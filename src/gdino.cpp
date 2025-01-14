#include "gdino.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

CGDINOModel::CGDINOModel(const char * encode, 
                         const char * decode, 
                         const char * vocab) {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "GDINO");

    std::ifstream f(vocab);
    if (f.is_open()) {
        int index = 0;
        for (std::string line; std::getline(f, line); ) {
            token_id_table[line] = index;
            id_token_table[index] = line;
            index++;
        }
    }

    special_token_ids = convert_tokens_to_ids({"[CLS]", "[SEP]", ".", "?"});

    encoder = std::make_unique<Ort::Session>(env, 
                                             encode, 
                                             Ort::SessionOptions(nullptr));

    decoder = std::make_unique<Ort::Session>(env, 
                                             decode, 
                                             Ort::SessionOptions(nullptr));
}

CGDINOModel::~CGDINOModel() {
    encoder->release();
    decoder->release();
}

std::vector<int> CGDINOModel::convert_tokens_to_ids(const std::vector<std::string>& tokens) {
    std::vector<int> input_ids;
    for (auto& token: tokens) {
        std::string sub_token = token, proc_token = sub_token;
        while (sub_token.size() > 0) {
            if (token_id_table.contains(sub_token)) {
                input_ids.emplace_back(token_id_table[sub_token]);
                if (sub_token == proc_token) break;

                sub_token = "##" + proc_token.substr(sub_token.size());
                proc_token = sub_token;
                continue;
            }
            sub_token = proc_token.substr(0, sub_token.size() - 1);
        }
    }

    return input_ids;
}

std::string CGDINOModel::convert_id_to_token(int id) {
    if (id > id_token_table.size()) return "";
    return id_token_table[id];
}

std::map<std::string, std::vector<int>> CGDINOModel::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;

    for (auto c: text) {
        if (c == ' ') {
            if (token.size() > 0) {
                tokens.emplace_back(token);
            }
            token = "";
            continue;
        }

        //punctuation
        if ((c >= 33 && c <= 47) || 
            (c >= 58 && c <= 64) || 
            (c >= 91 && c <= 96) || 
            (c >= 123 && c <= 126)) {
            tokens.emplace_back(token);

            token = c;
            tokens.emplace_back(token);

            token = "";
            continue;    
        }

        token += c;
    }

//    for (auto& t: tokens) std::cout << t << std::endl;
    tokens.insert(tokens.begin(), "[CLS]"); //101
    tokens.emplace_back("[SEP]"); //102

    std::vector<int> input_ids = convert_tokens_to_ids(tokens);

    int input_ids_len = input_ids.size();
    std::vector<int> attention_mask(input_ids_len);
    std::vector<int> token_type_ids(input_ids_len);
    std::vector<int> position_ids(input_ids_len);

    std::vector<int> special_token_indices;
    for (int i=0, sep=0, pos=0; 
         i<input_ids.size(); 
         ++i) {
        attention_mask[i] = ((input_ids[i] == 0) ? 0 : 1); //[PAD]
        token_type_ids[i] = sep;
        if (input_ids[i] == 102) sep++; //[SEP]

        if (std::find(special_token_ids.cbegin(), 
                      special_token_ids.cend(), 
                      input_ids[i]) != special_token_ids.cend()) {
            position_ids[i] = pos;
            pos = 0;

            special_token_indices.emplace_back(i);
        } else {
            position_ids[i] = pos;
            pos++;
        }
    }

    std::vector<int> text_self_attention_masks(input_ids_len * input_ids_len, 0);
    int prev_special_token_index = special_token_indices[0];
    for (auto& special_token_index: special_token_indices) {
        if (special_token_index == 0 || special_token_index == (input_ids_len - 1)) {
            text_self_attention_masks[special_token_index * input_ids_len + special_token_index] = 1;
            continue;
        }

        for (int i=(prev_special_token_index + 1); i<=special_token_index; ++i) {
            for (int j=(prev_special_token_index + 1); j<=special_token_index; ++j) {
                text_self_attention_masks[i * input_ids_len + j] = 1;
            }
        }
        prev_special_token_index = special_token_index;
    }

    std::map<std::string, std::vector<int>> tokenized;

    tokenized["input_ids"] = input_ids;
    tokenized["attention_mask"] = attention_mask;
    tokenized["token_type_ids"] = token_type_ids;
    tokenized["position_ids"] = position_ids;
    tokenized["text_self_attention_masks"] = text_self_attention_masks;

    return tokenized;
}

std::string CGDINOModel::pre_process_caption(const std::string& caption) {
    int begin = caption.find_first_not_of(' ');
    int end = caption.find_last_not_of(' ');
    std::string proc_caption = caption.substr(begin, end - begin + 1);
    std::transform(proc_caption.cbegin(), 
                   proc_caption.cend(), 
                   proc_caption.begin(), 
                   [](char c) { return std::tolower(c); });
    if (!proc_caption.ends_with('.')) proc_caption += '.';

    return proc_caption;
}

cv::Mat CGDINOModel::pre_process_image(const cv::Mat& image) {
    cv::Mat proc_image;
    cv::resize(image, proc_image, gdino_image_size);
    cv::cvtColor(proc_image, proc_image, cv::COLOR_BGR2RGB);
    proc_image.convertTo(proc_image, CV_32F, 1.0 / 255.0);
    proc_image = (proc_image - cv::Scalar(0.485, 0.456, 0.406)) / 
                 cv::Scalar(0.229, 0.224, 0.225);
    return cv::dnn::blobFromImage(proc_image);
}

int CGDINOModel::encode(const std::string& caption) {
    std::string proc_caption = pre_process_caption(caption);
    tokenized = tokenize(proc_caption);

    int input_ids_len = tokenized["input_ids"].size();

#if 0
    std::cout << "input_ids: " << std::endl;
    for (auto& input_id: tokenized["input_ids"]) {
        std::cout << input_id << " ";
    }
    std::cout << std::endl;

    std::cout << "attention_mask: " << std::endl;
    for (auto& attention_mask: tokenized["attention_mask"]) {
        std::cout << attention_mask << " ";
    }
    std::cout << std::endl;

    std::cout << "token_type_ids: " << std::endl;
    for (auto& token_type_id: tokenized["token_type_ids"]) {
        std::cout << token_type_id << " ";
    }
    std::cout << std::endl;

    std::cout << "position_ids: " << std::endl;
    for (auto& position_id: tokenized["position_ids"]) {
        std::cout << position_id << " ";
    }
    std::cout << std::endl;

    std::cout << "text_self_attention_masks: " << std::endl;
    for (int i=0; i<tokenized["text_self_attention_masks"].size(); ++i) {
        if (i && (i % input_ids_len == 0)) std::cout << std::endl;
        std::cout << tokenized["text_self_attention_masks"][i] << " ";
    }
    std::cout << std::endl;
#endif

    const std::vector<int64_t> shape_input_ids = { 1, input_ids_len };
    Ort::Value tensor_input_ids = Ort::Value::CreateTensor<int>(memory_info, 
                                                        tokenized["input_ids"].data(), 
                                                        tokenized["input_ids"].size(), 
                                                        shape_input_ids.data(), 
                                                        shape_input_ids.size());

    const std::vector<int64_t> shape_token_type_ids = { 1, input_ids_len };
    Ort::Value tensor_token_type_ids = Ort::Value::CreateTensor<int>(memory_info, 
                                                        tokenized["token_type_ids"].data(), 
                                                        tokenized["token_type_ids"].size(), 
                                                        shape_token_type_ids.data(), 
                                                        shape_token_type_ids.size());

    const std::vector<int64_t> shape_text_self_attention_masks = { 1, input_ids_len, input_ids_len };
    std::vector<uint8_t> text_self_attention_masks(tokenized["text_self_attention_masks"].size());
    std::transform(tokenized["text_self_attention_masks"].cbegin(), 
                   tokenized["text_self_attention_masks"].cend(), 
                   text_self_attention_masks.begin(), 
                   [](int c) { return static_cast<uint8_t>(c); });
    Ort::Value tensor_text_self_attention_masks = Ort::Value::CreateTensor<uint8_t>(memory_info, 
                                                        text_self_attention_masks.data(), 
                                                        text_self_attention_masks.size(), 
                                                        shape_text_self_attention_masks.data(), 
                                                        shape_text_self_attention_masks.size());

    const std::vector<int64_t> shape_position_ids = { 1, input_ids_len };
    Ort::Value tensor_position_ids = Ort::Value::CreateTensor<int>(memory_info, 
                                                        tokenized["position_ids"].data(), 
                                                        tokenized["position_ids"].size(), 
                                                        shape_position_ids.data(), 
                                                        shape_position_ids.size());
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(tensor_input_ids));
    inputs.emplace_back(std::move(tensor_token_type_ids));
    inputs.emplace_back(std::move(tensor_text_self_attention_masks));
    inputs.emplace_back(std::move(tensor_position_ids));

    auto outputs = encoder->Run(Ort::RunOptions(nullptr), 
                                  encode_input_names.data(), 
                                  inputs.data(), 
                                  inputs.size(), 
                                  encode_output_names.data(), 
                                  encode_output_names.size());

    tensor_last_hidden_state = std::move(outputs[0]);
#if 0    
    std::cout << "last_hidden_state shape:" << std::endl;
    for (auto& e: tensor_last_hidden_state.GetTensorTypeAndShapeInfo().GetShape()) {
        std::cout << e << " ";
    }
    std::cout << std::endl;
#endif

    return 0;
}

int CGDINOModel::decode(const cv::Mat& image, std::vector<std::tuple<float, std::string, cv::Rect>>& results) {
    cv::Mat blob = pre_process_image(image);
    const std::vector<int64_t> shape_image = { 1, 3, gdino_image_size.height, gdino_image_size.width };
    Ort::Value tensor_image = Ort::Value::CreateTensor<float>(memory_info, 
                                                             reinterpret_cast<float *>(blob.data), 
                                                             blob.total(), 
                                                             shape_image.data(), 
                                                             shape_image.size());

    std::vector<int> input_ids = tokenized["input_ids"];
    int input_ids_len = input_ids.size();

    const std::vector<int64_t> shape_attention_mask = { 1, input_ids_len };
    std::vector<uint8_t> attention_mask(tokenized["attention_mask"].size());
    std::transform(tokenized["attention_mask"].cbegin(), 
                   tokenized["attention_mask"].cend(), 
                   attention_mask.begin(), 
                   [](int c) { return static_cast<uint8_t>(c); });
    Ort::Value tensor_attention_mask = Ort::Value::CreateTensor<uint8_t>(memory_info, 
                                                        attention_mask.data(), 
                                                        attention_mask.size(), 
                                                        shape_attention_mask.data(), 
                                                        shape_attention_mask.size());

    const std::vector<int64_t> shape_position_ids = { 1, input_ids_len };
    Ort::Value tensor_position_ids = Ort::Value::CreateTensor<int>(memory_info, 
                                                        tokenized["position_ids"].data(), 
                                                        tokenized["position_ids"].size(), 
                                                        shape_position_ids.data(), 
                                                        shape_position_ids.size());

    const std::vector<int64_t> shape_text_self_attention_masks = { 1, input_ids_len, input_ids_len };
    std::vector<uint8_t> text_self_attention_masks(tokenized["text_self_attention_masks"].size());
    std::transform(tokenized["text_self_attention_masks"].cbegin(), 
                   tokenized["text_self_attention_masks"].cend(), 
                   text_self_attention_masks.begin(), 
                   [](int c) { return static_cast<uint8_t>(c); });
    Ort::Value tensor_text_self_attention_masks = Ort::Value::CreateTensor<uint8_t>(memory_info, 
                                                        text_self_attention_masks.data(), 
                                                        text_self_attention_masks.size(), 
                                                        shape_text_self_attention_masks.data(), 
                                                        shape_text_self_attention_masks.size());

    const std::vector<int64_t> shape_box_threshold = { 1 };
    Ort::Value tensor_box_threshold = Ort::Value::CreateTensor<float>(memory_info, 
                                                                     const_cast<float *>(&box_threshold), 
                                                                     1, 
                                                                     shape_box_threshold.data(), 
                                                                     shape_box_threshold.size());
    const std::vector<int64_t> shape_text_threshold = { 1 };
    Ort::Value tensor_text_threshold = Ort::Value::CreateTensor<float>(memory_info, 
                                                                     const_cast<float *>(&text_threshold), 
                                                                     1, 
                                                                     shape_text_threshold.data(), 
                                                                     shape_text_threshold.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(tensor_image));
    inputs.emplace_back(std::move(tensor_last_hidden_state));
    inputs.emplace_back(std::move(tensor_attention_mask));
    inputs.emplace_back(std::move(tensor_position_ids));
    inputs.emplace_back(std::move(tensor_text_self_attention_masks));
    inputs.emplace_back(std::move(tensor_box_threshold));
    inputs.emplace_back(std::move(tensor_text_threshold));

    auto outputs = decoder->Run(Ort::RunOptions(nullptr), 
                                decode_input_names.data(), 
                                inputs.data(), 
                                inputs.size(), 
                                decode_output_names.data(), 
                                decode_output_names.size());
    Ort::Value predict_logits = std::move(outputs[0]);
    Ort::Value predict_boxes = std::move(outputs[1]);
    Ort::Value predict_masks = std::move(outputs[2]);

    int predict_num = predict_logits.GetTensorTypeAndShapeInfo().GetShape()[1];

    const float * predict_logits_data = predict_logits.GetTensorData<float>();
    const float * predict_boxes_data = predict_boxes.GetTensorData<float>();
    const bool * predict_masks_data = predict_masks.GetTensorData<bool>();

    for (int i=0; i<predict_num; ++i) {
        //score
        float score = predict_logits_data[i];

        //caption
        std::string caption = "";
        for (int j=0; j<input_ids_len; ++j) {
            if (predict_masks_data[i * 256 + j]) {
                std::string token = convert_id_to_token(input_ids[j]);
                if (token.starts_with("##")) token = token.substr(2);
                caption += token;
            }
        }

        //box
        float cx = predict_boxes_data[i * 4];
        float cy = predict_boxes_data[i * 4 + 1];
        float w = predict_boxes_data[i * 4 + 2];
        float h = predict_boxes_data[i * 4 + 3];

        float x1 = (cx - w / 2) * image.cols;
        float y1 = (cy - h / 2) * image.rows;
        float x2 = (cx + w / 2) * image.cols;
        float y2 = (cy + h / 2) * image.rows;

        cv::Rect box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));

        results.emplace_back(std::make_tuple(score, caption, box));
    }

    return 0;
}

int CGDINOModel::process(const cv::Mat& image, 
                         const std::string& caption, 
                         std::vector<std::tuple<float, std::string, cv::Rect>>& results) {
    encode(caption);
    decode(image, results);
    return 0;
}
