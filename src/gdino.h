#pragma once

#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <tuple>
#include <vector>

class CGDINOModel {
    const cv::Size gdino_image_size = { 800, 800 };
    const float box_threshold = 0.35f;
    const float text_threshold = 0.25f;

    public:
        CGDINOModel(const char * encode, 
                    const char * decode, 
                    const char * vocab);
        ~CGDINOModel();

    private:
        std::vector<int> convert_tokens_to_ids(const std::vector<std::string>& tokens);
        std::string convert_id_to_token(int id);
        std::map<std::string, std::vector<int>> tokenize(const std::string& text);
        std::string pre_process_caption(const std::string& caption);
        cv::Mat pre_process_image(const cv::Mat& image);

        int encode(const std::string& caption);
        int decode(const cv::Mat& image, std::vector<std::tuple<float, std::string, cv::Rect>>& results);

    public:
        int process(const cv::Mat& image, 
                    const std::string& caption, 
                    std::vector<std::tuple<float, std::string, cv::Rect>>& results);

    private:
        std::map<std::string, int> token_id_table;
        std::map<int, std::string> id_token_table;

        std::vector<int> special_token_ids;
        std::map<std::string, std::vector<int>> tokenized;

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::unique_ptr<Ort::Session> encoder;
        const std::vector<const char *> encode_input_names = {
            "input_ids", "token_type_ids", "text_self_attention_masks", "position_ids"
        };
        const std::vector<const char *> encode_output_names = {
            "last_hidden_state"
        };
        Ort::Value tensor_last_hidden_state = Ort::Value(nullptr);

        std::unique_ptr<Ort::Session> decoder;
        const std::vector<const char *> decode_input_names = {
            "image", "last_hidden_state", "attention_mask", 
            "position_ids", "text_self_attention_masks", 
            "box_threshold", "text_threshold"
        };
        const std::vector<const char *> decode_output_names = {
            "logits", "boxes", "masks"
        };
};