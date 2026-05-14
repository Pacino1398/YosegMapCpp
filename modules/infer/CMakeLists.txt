add_library(yoseg_infer
  src/infer_factory.cpp
  src/rknn_engine.cpp
  src/rknn_runtime_adapter.cpp
  src/yolov5_seg_postprocessor.cpp
)

target_include_directories(yoseg_infer PUBLIC include)
target_link_libraries(yoseg_infer PUBLIC yoseg_core)

find_package(OpenCV QUIET COMPONENTS core imgproc)
if(OpenCV_FOUND)
  target_compile_definitions(yoseg_infer PRIVATE YOSEG_INFER_HAS_OPENCV=1)
  target_link_libraries(yoseg_infer PUBLIC ${OpenCV_LIBS})
endif()

if(YOSEG_WITH_RKNN)
  target_compile_definitions(yoseg_infer PRIVATE YOSEG_WITH_RKNN_RUNTIME=1)
endif()
