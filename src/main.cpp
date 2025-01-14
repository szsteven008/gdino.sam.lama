#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

#include "gdino.h"
#include "sam2.h"
#include "lama.h"

namespace po = boost::program_options;

int main(int argc, char * argv[]) {
    po::options_description opts("Allowed options");
    opts.add_options()
                    ("help", "help message")
                    ("image,i", po::value<std::string>(), "input image")
                    ("prompt,p", po::value<std::string>(), "prompt")
                    ("kernel,k", po::value<int>()->default_value(9), "kernel size")
                    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    if (vm.count("help") > 0 || 
        vm.count("image") == 0 || 
        vm.count("prompt") == 0) {
        std::cout << opts << std::endl;
        return 0;
    }

    cv::Mat image = cv::imread(vm["image"].as<std::string>());
    std::string caption = vm["prompt"].as<std::string>();

    CGDINOModel gdino("models/gdino.encoder.onnx", 
                      "models/gdino.decoder.onnx", 
                      "models/vocab.txt");
    CSam2 sam2("models/sam2_1.encoder.onnx", 
               "models/sam2_1.decoder.box.onnx");
    CLama lama("models/lama.onnx", vm["kernel"].as<int>());

    std::vector<std::tuple<float, std::string, cv::Rect>> results;
    gdino.process(image, caption, results);

    std::vector<cv::Rect> boxes;
    cv::Mat result_image = image.clone();
    for (auto& result: results) {
        float score;
        std::string caption;
        cv::Rect box;

        tie(score, caption, box) = result;

        boxes.emplace_back(box);

        cv::rectangle(result_image, box, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("image", result_image);
    cv::waitKey();

    cv::Mat mask;
    sam2.process(image, boxes, mask);

    cv::cvtColor(result_image, result_image, cv::COLOR_BGR2BGRA);

    cv::Mat result_mask;
    cv::cvtColor(mask, result_mask, cv::COLOR_GRAY2BGRA);
    result_mask.setTo(cv::Scalar(0, 203, 255, static_cast<uchar>(255 * 0.73)), 
                       (result_mask > 128));
    cv::addWeighted(result_image, 1.0, result_mask, 0.3, 0.0, result_image);
    cv::imshow("image", result_image);
    cv::waitKey();


    lama.inpainting(image, mask, result_image);

    cv::imshow("image", result_image);
    cv::waitKey();

    cv::destroyAllWindows();

    return 0;
}