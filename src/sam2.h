#pragma once

#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>

class CSam2 {
        const cv::Size sam2_image_size = { 1024, 1024 };

    public:
        CSam2(const char * encode, const char * decode);
        ~CSam2();

    private:
        int encode(const cv::Mat& image);
        int decode(const cv::Mat& image, 
                   const std::vector<cv::Rect> boxes, 
                   cv::Mat& mask);

    public:
        int process(const cv::Mat& image, 
                    const std::vector<cv::Rect> boxes, 
                    cv::Mat& mask);

    private:
        std::unique_ptr<Ort::Session> encoder;
        std::unique_ptr<Ort::Session> decoder;

        Ort::Value tensor_image_embeddings = Ort::Value(nullptr);
        Ort::Value tensor_high_res_features1 = Ort::Value(nullptr);
        Ort::Value tensor_high_res_features2 = Ort::Value(nullptr);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, 
                                                                 OrtMemTypeCPU);
        const std::vector<const char *> encode_input_names = {
            "image"
        };
        const std::vector<const char *> encode_output_names = {
            "image_embeddings", "high_res_feats1", "high_res_feats2"
        };

        const std::vector<const char *> decode_input_names = {
            "image_embeddings", "high_res_features1", "high_res_features2", 
            "boxes"
        };
        const std::vector<const char *> decode_output_names = {
            "masks", "iou_predictions"
        };
};