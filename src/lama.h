#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>

class CLama {
    const cv::Size lama_image_size = { 512, 512 };

    public:
        CLama(const char * model, int kernel = 9);
        ~CLama();

    public:
        int inpainting(const cv::Mat& image, const cv::Mat& mask, cv::Mat& result);

    private:
        std::unique_ptr<Ort::Session> session;

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, 
                                                                 OrtMemTypeCPU);
        
        const std::vector<const char *> input_names = { "image", "mask" };
        const std::vector<const char *> output_names = { "output" };

        cv::Size dilate_kernel;
};