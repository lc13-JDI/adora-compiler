add_llvm_executable(cstore-dfg-contract-test
  cgra-mapper/cstore_contract/dfg_contract_unit.cpp)
target_include_directories(cstore-dfg-contract-test PRIVATE
  ${CMAKE_SOURCE_DIR}/mapper/include
  ${CMAKE_SOURCE_DIR}/lib/DFG/inc)
target_compile_definitions(cstore-dfg-contract-test PRIVATE
  SPDLOG_NO_EXCEPTIONS)
target_link_libraries(cstore-dfg-contract-test PRIVATE
  CGRAMapperLIB
  GeneralCDFGGen
  MLIRADORAOps
  MLIRADORAKernelOp)
llvm_update_compile_flags(cstore-dfg-contract-test)

add_llvm_executable(cstore-operand-construction-test
  cgra-opt/cdfggen/cstore_operand_construction_unit.cpp)
target_include_directories(cstore-operand-construction-test PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/DFG/inc)
target_link_libraries(cstore-operand-construction-test PRIVATE
  GeneralCDFGGen
  MLIRADORAOps
  MLIRADORAKernelOp)
llvm_update_compile_flags(cstore-operand-construction-test)

set(ADORA_TEST_DEPENDS
  cgra-opt
  tensor-opt
  cgra-mapper
  cstore-dfg-contract-test
  cstore-operand-construction-test
  FileCheck
)

set(ADORA_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ADORA_TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(ADORA_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

set(ADORA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ADORA_BINARY_DIR ${CMAKE_BINARY_DIR})

# where your built tool is
set(ADORA_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-adora "Running the ADORA regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ADORA_TEST_DEPENDS}
)


# LLVM's recursive helper creates a target for every directory, including
# directories without lit tests. Reserve test/rtl for the optional cocotb suite.
if(NOT LLVM_ENABLE_IDE)
  file(GLOB suite_dirs RELATIVE "${ADORA_TEST_DIR}" "${ADORA_TEST_DIR}/*")
  foreach(suite IN LISTS suite_dirs)
    set(suite_path "${ADORA_TEST_DIR}/${suite}")
    if(IS_DIRECTORY "${suite_path}" AND NOT suite STREQUAL "rtl"
       AND NOT suite MATCHES "Inputs|Output")
      string(TOLOWER "ADORA-${suite}" suite_name)
      add_lit_target("check-${suite_name}" "Running lit suite ${suite_path}"
        "${suite_path}" DEPENDS ${ADORA_TEST_DEPENDS})
      add_lit_testsuites("${suite_name}" "${suite_path}" DEPENDS ${ADORA_TEST_DEPENDS})
    endif()
  endforeach()
endif()
