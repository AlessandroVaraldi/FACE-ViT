cmake_minimum_required(VERSION 3.18)

project(vit C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

function(embed_binary SRC_FILE SYMBOL OUT_VAR)
    get_filename_component(ABS_PATH "${SRC_FILE}" ABSOLUTE)
    get_filename_component(FILE_NAME "${SRC_FILE}" NAME)

    set(TMP_OBJ   "${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.tmp.o")
    set(FINAL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.o")

    add_custom_command(
        OUTPUT  ${TMP_OBJ}
        COMMAND ${CMAKE_OBJCOPY}
                -I binary          # input  : raw data
                -O elf64-x86-64    # output : relocatable ELF
                -B i386            # neutral machine type
                "${ABS_PATH}" "${TMP_OBJ}"
        DEPENDS "${ABS_PATH}"
        COMMENT "Embedding ${FILE_NAME} (raw -> ELF)"
        VERBATIM
    )

    file(TO_CMAKE_PATH "${ABS_PATH}" ABS_UNIX)
    string(REGEX REPLACE "^/" ""      PATH_NO_ROOT "${ABS_UNIX}")
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" SYM_BASE   "${PATH_NO_ROOT}")

    add_custom_command(
        OUTPUT  ${FINAL_OBJ}
        COMMAND ${CMAKE_OBJCOPY}
                --redefine-sym _binary__${SYM_BASE}_start=${SYMBOL}
                --strip-symbol  _binary__${SYM_BASE}_end
                --strip-symbol  _binary__${SYM_BASE}_size
                ${TMP_OBJ} ${FINAL_OBJ}
        DEPENDS ${TMP_OBJ}
        COMMENT "Renaming symbol -> ${SYMBOL}"
        VERBATIM
    )

    set(${OUT_VAR} "${FINAL_OBJ}" PARENT_SCOPE)
endfunction()

embed_binary("${PROJECT_SOURCE_DIR}/include/weights.bin" weights_bin  W_OBJ)

file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.c")

add_executable(alem ${SRC_FILES} ${W_OBJ})
target_include_directories(alem PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(alem PRIVATE m)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(alem PRIVATE
        -Wall -Wextra -Wconversion -Wno-sign-conversion -pipe
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        $<$<CONFIG:Debug>:-fsanitize=address -g>
    )
    target_link_options(alem PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
endif()

# Optional: build PIE outside Debug
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(alem PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
