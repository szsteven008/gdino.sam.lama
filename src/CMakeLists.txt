set(FILES main.cpp gdino.cpp sam2.cpp lama.cpp)

add_executable(main ${FILES})
target_link_libraries(main ${OpenCV_LIBS} ${Boost_LIBRARIES} onnx onnxruntime)