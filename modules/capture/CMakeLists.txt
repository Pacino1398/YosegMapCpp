add_library(yoseg_capture
  src/capture_stub.cpp
)

target_include_directories(yoseg_capture PUBLIC include)

if(YOSEG_WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
  if(OpenCV_FOUND)
    target_compile_definitions(yoseg_capture PRIVATE YOSEG_HAS_OPENCV=1)
    target_link_libraries(yoseg_capture PUBLIC ${OpenCV_LIBS})
    message(STATUS "yoseg_capture: OpenCV enabled")
  else()
    message(STATUS "yoseg_capture: OpenCV not found, fallback to synthetic frames")
  endif()
endif()
