project(colonies3D)

#Can manually add the sources using the set command as follows:
set(SOURCES colonies3D.hpp colonies3D.cpp example3D.cpp)

#However, the file(GLOB...) allows for wildcard additions:

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
add_executable(example3D ${SOURCES})
target_link_libraries(example3D ${ARMADILLO_LIBRARIES})
