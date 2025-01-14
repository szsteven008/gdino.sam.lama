#include "lama.h"
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

CLama::CLama(const char * model, int kernel /* = 9 */) : 
    dilate_kernel(kernel, kernel) {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "LAMA");

    session = std::make_unique<Ort::Session>(env, 
                                  model, 
                                  Ort::SessionOptions(nullptr));
}

CLama::~CLama() {
    session->release();
}

int CLama::inpainting(const cv::Mat& image, const cv::Mat& mask, cv::Mat& result) {
    cv::Mat proc_image, proc_mask, blob_image, blob_mask;
    cv::resize(image, proc_image, lama_image_size);
    cv::resize(mask, proc_mask, lama_image_size);

    cv::cvtColor(proc_image, proc_image, cv::COLOR_BGR2RGB);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, dilate_kernel);
    cv::dilate(proc_mask, proc_mask, kernel);
    cv::threshold(proc_mask, proc_mask, 127, 255, cv::THRESH_BINARY);

    proc_image.convertTo(proc_image, CV_32F, 1.0 / 255.0);
    proc_mask.convertTo(proc_mask, CV_32FC1, 1.0 / 255.0);

    blob_image = cv::dnn::blobFromImage(proc_image);
    blob_mask = cv::dnn::blobFromImage(proc_mask);

    std::vector<int64_t> image_shape = { 1, 3, lama_image_size.height, lama_image_size.width };
    Ort::Value tensor_image = Ort::Value::CreateTensor<float>(memory_info, 
                                                              reinterpret_cast<float *>(blob_image.data), 
                                                              blob_image.total(), 
                                                              image_shape.data(), 
                                                              image_shape.size());
    std::vector<int64_t> mask_shape = { 1, 1, lama_image_size.height, lama_image_size.width };
    Ort::Value tensor_mask = Ort::Value::CreateTensor<float>(memory_info, 
                                                              reinterpret_cast<float *>(blob_mask.data), 
                                                              blob_mask.total(), 
                                                              mask_shape.data(), 
                                                              mask_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(tensor_image));
    inputs.emplace_back(std::move(tensor_mask));

    auto outputs = session->Run(Ort::RunOptions(nullptr), 
                                input_names.data(), 
                                inputs.data(), 
                                inputs.size(), 
                                output_names.data(), 
                                output_names.size());

    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const uint8_t * output_data = outputs[0].GetTensorData<uint8_t>();
    int output_data_area = output_shape[2] * output_shape[3];
    std::vector<cv::Mat> channels(output_shape[1]);
    for (int i=0; i<channels.size(); ++i) {
        channels[i] = cv::Mat(output_shape[2], 
                              output_shape[3], 
                              CV_8UC1, 
                              (void *)(output_data + i * output_data_area));
    }

    cv::Mat output_image;
    cv::merge(channels, output_image);

    cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
    cv::resize(output_image, result, image.size());

    return 0;
}
