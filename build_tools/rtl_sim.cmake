find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ADORA_RTL_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with cocotb 1.9 and numpy")
set(ADORA_RTL_ASSETS "${CMAKE_BINARY_DIR}/rtl-assets" CACHE PATH "Offline hardware asset store")
set(ADORA_RTL_CACHE "${CMAKE_BINARY_DIR}/rtl-models" CACHE PATH "Directory for verified model builds")
set(ADORA_RTL_JOBS 2 CACHE STRING "Parallel C++ compilation jobs per RTL model")
set(ADORA_RTL_CGEIST "" CACHE FILEPATH "Optional Polygeist cgeist executable")
set(rtl_args
  --assets "${ADORA_RTL_ASSETS}" --cache "${ADORA_RTL_CACHE}"
  --output-root "${CMAKE_BINARY_DIR}/rtl-results" --build-dir "${CMAKE_BINARY_DIR}"
  --jobs "${ADORA_RTL_JOBS}")
if(ADORA_RTL_CGEIST)
  list(APPEND rtl_args --cgeist "${ADORA_RTL_CGEIST}")
endif()
set(rtl_runner "${ADORA_RTL_PYTHON}" -I "${PROJECT_SOURCE_DIR}/build_tools/rtl_sim.py" run)
add_custom_target(check-adora-rtl
  COMMAND ${rtl_runner} --case short-tail ${rtl_args}
  COMMAND ${rtl_runner} --case canvas-mask ${rtl_args}
  COMMAND ${rtl_runner} --case vector-add ${rtl_args}
  COMMAND ${rtl_runner} --case tcp-client ${rtl_args}
  DEPENDS cgra-opt cgra-mapper
  USES_TERMINAL VERBATIM)
add_custom_target(check-adora-rtl-ownership
  COMMAND ${rtl_runner} --case ownership ${rtl_args}
  DEPENDS cgra-opt cgra-mapper
  USES_TERMINAL VERBATIM)
