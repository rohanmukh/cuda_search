# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/rm38/Downloads/clion-2019.3.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/rm38/Downloads/clion-2019.3.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rm38/CLionProjects/cuda_code_search

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rm38/CLionProjects/cuda_code_search/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/cuda_code_search.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_code_search.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_code_search.dir/flags.make

CMakeFiles/cuda_code_search.dir/serial_code.cpp.o: CMakeFiles/cuda_code_search.dir/flags.make
CMakeFiles/cuda_code_search.dir/serial_code.cpp.o: ../serial_code.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rm38/CLionProjects/cuda_code_search/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cuda_code_search.dir/serial_code.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cuda_code_search.dir/serial_code.cpp.o -c /home/rm38/CLionProjects/cuda_code_search/serial_code.cpp

CMakeFiles/cuda_code_search.dir/serial_code.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuda_code_search.dir/serial_code.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rm38/CLionProjects/cuda_code_search/serial_code.cpp > CMakeFiles/cuda_code_search.dir/serial_code.cpp.i

CMakeFiles/cuda_code_search.dir/serial_code.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuda_code_search.dir/serial_code.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rm38/CLionProjects/cuda_code_search/serial_code.cpp -o CMakeFiles/cuda_code_search.dir/serial_code.cpp.s

CMakeFiles/cuda_code_search.dir/mem_error.cpp.o: CMakeFiles/cuda_code_search.dir/flags.make
CMakeFiles/cuda_code_search.dir/mem_error.cpp.o: ../mem_error.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rm38/CLionProjects/cuda_code_search/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cuda_code_search.dir/mem_error.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cuda_code_search.dir/mem_error.cpp.o -c /home/rm38/CLionProjects/cuda_code_search/mem_error.cpp

CMakeFiles/cuda_code_search.dir/mem_error.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuda_code_search.dir/mem_error.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rm38/CLionProjects/cuda_code_search/mem_error.cpp > CMakeFiles/cuda_code_search.dir/mem_error.cpp.i

CMakeFiles/cuda_code_search.dir/mem_error.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuda_code_search.dir/mem_error.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rm38/CLionProjects/cuda_code_search/mem_error.cpp -o CMakeFiles/cuda_code_search.dir/mem_error.cpp.s

# Object files for target cuda_code_search
cuda_code_search_OBJECTS = \
"CMakeFiles/cuda_code_search.dir/serial_code.cpp.o" \
"CMakeFiles/cuda_code_search.dir/mem_error.cpp.o"

# External object files for target cuda_code_search
cuda_code_search_EXTERNAL_OBJECTS =

cuda_code_search: CMakeFiles/cuda_code_search.dir/serial_code.cpp.o
cuda_code_search: CMakeFiles/cuda_code_search.dir/mem_error.cpp.o
cuda_code_search: CMakeFiles/cuda_code_search.dir/build.make
cuda_code_search: CMakeFiles/cuda_code_search.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rm38/CLionProjects/cuda_code_search/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable cuda_code_search"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_code_search.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_code_search.dir/build: cuda_code_search

.PHONY : CMakeFiles/cuda_code_search.dir/build

CMakeFiles/cuda_code_search.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_code_search.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_code_search.dir/clean

CMakeFiles/cuda_code_search.dir/depend:
	cd /home/rm38/CLionProjects/cuda_code_search/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rm38/CLionProjects/cuda_code_search /home/rm38/CLionProjects/cuda_code_search /home/rm38/CLionProjects/cuda_code_search/cmake-build-debug /home/rm38/CLionProjects/cuda_code_search/cmake-build-debug /home/rm38/CLionProjects/cuda_code_search/cmake-build-debug/CMakeFiles/cuda_code_search.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda_code_search.dir/depend

