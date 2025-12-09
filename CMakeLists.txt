cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)

project(diabetes-detection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

# Adiciona a subpasta da biblioteca
add_subdirectory(pico-tflmicro) 

add_executable(diabetes-detection 
    diabetes-detection.c 
    tinyml.cpp
)

pico_set_program_name(diabetes-detection "diabetes-detection")
pico_set_program_version(diabetes-detection "0.1")

pico_enable_stdio_uart(diabetes-detection 0)
pico_enable_stdio_usb(diabetes-detection 1)

# --- A CORREÇÃO ESTÁ AQUI EMBAIXO ---
target_link_libraries(diabetes-detection 
    pico_stdlib
    hardware_adc
    pico-tflmicro  # <--- Mudei de underline (_) para hífen (-)
)

target_include_directories(diabetes-detection PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/pico-tflmicro/src
)

pico_add_extra_outputs(diabetes-detection)