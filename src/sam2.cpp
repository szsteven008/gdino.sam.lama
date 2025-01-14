#include "sam2.h"
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

CSam2::CSam2(const char * encode, const char * decode) {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "SAM2");

    encoder = std::make_unique<Ort::Session>(env, 
                                             encode, 
                                             Ort::SessionOptions(nullptr));

    decoder = std::make_unique<Ort::Session>(env, 
                                             decode, 
                                             Ort::SessionOptions(nullptr));
}

CSam2::~CSam2() {
    encoder->release();
    decoder->release();
}

int CSam2::encode(const cv::Mat& image) {
    cv::Mat proc_image;
    cv::resize(image, proc_image, sam2_image_size);
    proc_image.convertTo(proc_image, CV_32F, 1.0 / 255.0);
    proc_image = (proc_image - cv::Scalar(0.485, 0.456, 0.406)) / 
                 cv::Scalar(0.229, 0.224, 0.225);
    cv::Mat blob = cv::dnn::blobFromImage(proc_image, 
                                          1.0, 
                                          proc_image.size(), 
                                          cv::Scalar(), 
                                          true);

    const std::vector<int64_t> shape_input = { 1, 
                                               3, 
                                               sam2_image_size.height, 
                                               sam2_image_size.width };
    Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, 
                                                       reinterpret_cast<float *>(blob.data), 
                                                       blob.total(), 
                                                       shape_input.data(), 
                                                       shape_input.size());

    auto outputs = encoder->Run(Ort::RunOptions(nullptr), 
                                encode_input_names.data(), 
                                &input, 
                                1, 
                                encode_output_names.data(), 
                                encode_output_names.size());

    tensor_image_embeddings = std::move(outputs[0]);
    tensor_high_res_features1 = std::move(outputs[1]);
    tensor_high_res_features2 = std::move(outputs[2]);

    return 0;
}

int CSam2::decode(const cv::Mat& image, 
                  const std::vector<cv::Rect> boxes, 
                  cv::Mat& mask) {
    std::vector<float> ratios = { 1.0f * sam2_image_size.width / image.cols, 
                                  1.0f * sam2_image_size.height / image.rows };

    std::vector<int32_t> boxes_data;
    for (auto& box: boxes) {
        int x1 = box.x * ratios[0];
        int y1 = box.y * ratios[1];
        int x2 = (box.x + box.width) * ratios[0];
        int y2 = (box.y + box.height) * ratios[1];

        boxes_data.emplace_back(x1);
        boxes_data.emplace_back(y1);
        boxes_data.emplace_back(x2);
        boxes_data.emplace_back(y2);
    }
    const std::vector<int64_t> boxes_shape = { static_cast<int64_t>(boxes.size()), 2, 2 };
    Ort::Value tensor_boxes = Ort::Value::CreateTensor<int>(memory_info, 
                                                            boxes_data.data(), 
                                                            boxes_data.size(), 
                                                            boxes_shape.data(), 
                                                            boxes_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(tensor_image_embeddings));
    inputs.emplace_back(std::move(tensor_high_res_features1));
    inputs.emplace_back(std::move(tensor_high_res_features2));
    inputs.emplace_back(std::move(tensor_boxes));

    auto outputs = decoder->Run(Ort::RunOptions(nullptr), 
                                decode_input_names.data(), 
                                inputs.data(), 
                                inputs.size(), 
                                decode_output_names.data(), 
                                decode_output_names.size());

    std::vector<int64_t> masks_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const uint8_t * masks_data = outputs[0].GetTensorData<uint8_t>();
    int mask_data_len = masks_shape[1] * masks_shape[2] * masks_shape[3];
    cv::Mat predict_mask_combine = cv::Mat::zeros(masks_shape[2], 
                                                  masks_shape[3], 
                                                  CV_8UC1);
    for (int i=0; i<masks_shape[0]; ++i) {
        cv::Mat predict_mask = cv::Mat(masks_shape[2], 
                                       masks_shape[3], 
                                       CV_8UC1, 
                                       (void *)(masks_data + i * mask_data_len));
        predict_mask_combine = predict_mask_combine + predict_mask;
    }

    predict_mask_combine.convertTo(predict_mask_combine, CV_8UC1, 255.0);
    cv::threshold(predict_mask_combine, 
                  mask, 
                  127, 
                  255, 
                  cv::THRESH_BINARY);
    cv::resize(mask, mask, image.size());

    return 0;
}

int CSam2::process(const cv::Mat& image, 
                   const std::vector<cv::Rect> boxes, 
                   cv::Mat& mask) {
    encode(image);
    decode(image, boxes, mask);
    return 0;
}
