# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /apps/all/CMake/3.16.4-GCCcore-9.3.0/bin/cmake

# The command to remove a file.
RM = /apps/all/CMake/3.16.4-GCCcore-9.3.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/training/dd-20-28-209/avs2/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/training/dd-20-28-209/avs2/build

# Include any dependencies generated for this target.
include CMakeFiles/PMC.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PMC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PMC.dir/flags.make

CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.o: CMakeFiles/PMC.dir/flags.make
CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.o: /home/training/dd-20-28-209/avs2/src/common/base_mesh_builder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/training/dd-20-28-209/avs2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.o"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.o -c /home/training/dd-20-28-209/avs2/src/common/base_mesh_builder.cpp

CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.i"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/training/dd-20-28-209/avs2/src/common/base_mesh_builder.cpp > CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.i

CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.s"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/training/dd-20-28-209/avs2/src/common/base_mesh_builder.cpp -o CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.s

CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.o: CMakeFiles/PMC.dir/flags.make
CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.o: /home/training/dd-20-28-209/avs2/src/common/parametric_scalar_field.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/training/dd-20-28-209/avs2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.o"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.o -c /home/training/dd-20-28-209/avs2/src/common/parametric_scalar_field.cpp

CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.i"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/training/dd-20-28-209/avs2/src/common/parametric_scalar_field.cpp > CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.i

CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.s"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/training/dd-20-28-209/avs2/src/common/parametric_scalar_field.cpp -o CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.s

CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.o: CMakeFiles/PMC.dir/flags.make
CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.o: /home/training/dd-20-28-209/avs2/src/common/ref_mesh_builder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/training/dd-20-28-209/avs2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.o"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.o -c /home/training/dd-20-28-209/avs2/src/common/ref_mesh_builder.cpp

CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.i"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/training/dd-20-28-209/avs2/src/common/ref_mesh_builder.cpp > CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.i

CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.s"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/training/dd-20-28-209/avs2/src/common/ref_mesh_builder.cpp -o CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.s

CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.o: CMakeFiles/PMC.dir/flags.make
CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.o: /home/training/dd-20-28-209/avs2/src/parallel_builder/loop_mesh_builder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/training/dd-20-28-209/avs2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.o"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.o -c /home/training/dd-20-28-209/avs2/src/parallel_builder/loop_mesh_builder.cpp

CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.i"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/training/dd-20-28-209/avs2/src/parallel_builder/loop_mesh_builder.cpp > CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.i

CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.s"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/training/dd-20-28-209/avs2/src/parallel_builder/loop_mesh_builder.cpp -o CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.s

CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.o: CMakeFiles/PMC.dir/flags.make
CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.o: /home/training/dd-20-28-209/avs2/src/parallel_builder/tree_mesh_builder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/training/dd-20-28-209/avs2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.o"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.o -c /home/training/dd-20-28-209/avs2/src/parallel_builder/tree_mesh_builder.cpp

CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.i"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/training/dd-20-28-209/avs2/src/parallel_builder/tree_mesh_builder.cpp > CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.i

CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.s"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/training/dd-20-28-209/avs2/src/parallel_builder/tree_mesh_builder.cpp -o CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.s

CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.o: CMakeFiles/PMC.dir/flags.make
CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.o: /home/training/dd-20-28-209/avs2/src/parallel_builder/cached_mesh_builder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/training/dd-20-28-209/avs2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.o"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.o -c /home/training/dd-20-28-209/avs2/src/parallel_builder/cached_mesh_builder.cpp

CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.i"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/training/dd-20-28-209/avs2/src/parallel_builder/cached_mesh_builder.cpp > CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.i

CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.s"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/training/dd-20-28-209/avs2/src/parallel_builder/cached_mesh_builder.cpp -o CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.s

CMakeFiles/PMC.dir/main.cpp.o: CMakeFiles/PMC.dir/flags.make
CMakeFiles/PMC.dir/main.cpp.o: /home/training/dd-20-28-209/avs2/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/training/dd-20-28-209/avs2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/PMC.dir/main.cpp.o"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PMC.dir/main.cpp.o -c /home/training/dd-20-28-209/avs2/src/main.cpp

CMakeFiles/PMC.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PMC.dir/main.cpp.i"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/training/dd-20-28-209/avs2/src/main.cpp > CMakeFiles/PMC.dir/main.cpp.i

CMakeFiles/PMC.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PMC.dir/main.cpp.s"
	/apps/all/iccifort/2020.1.217/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/training/dd-20-28-209/avs2/src/main.cpp -o CMakeFiles/PMC.dir/main.cpp.s

# Object files for target PMC
PMC_OBJECTS = \
"CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.o" \
"CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.o" \
"CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.o" \
"CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.o" \
"CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.o" \
"CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.o" \
"CMakeFiles/PMC.dir/main.cpp.o"

# External object files for target PMC
PMC_EXTERNAL_OBJECTS =

PMC: CMakeFiles/PMC.dir/common/base_mesh_builder.cpp.o
PMC: CMakeFiles/PMC.dir/common/parametric_scalar_field.cpp.o
PMC: CMakeFiles/PMC.dir/common/ref_mesh_builder.cpp.o
PMC: CMakeFiles/PMC.dir/parallel_builder/loop_mesh_builder.cpp.o
PMC: CMakeFiles/PMC.dir/parallel_builder/tree_mesh_builder.cpp.o
PMC: CMakeFiles/PMC.dir/parallel_builder/cached_mesh_builder.cpp.o
PMC: CMakeFiles/PMC.dir/main.cpp.o
PMC: CMakeFiles/PMC.dir/build.make
PMC: /apps/all/imkl/2020.1.217-iimpi-2020a/lib/intel64/libiomp5.so
PMC: /lib64/libpthread.so
PMC: CMakeFiles/PMC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/training/dd-20-28-209/avs2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable PMC"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PMC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PMC.dir/build: PMC

.PHONY : CMakeFiles/PMC.dir/build

CMakeFiles/PMC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PMC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PMC.dir/clean

CMakeFiles/PMC.dir/depend:
	cd /home/training/dd-20-28-209/avs2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/training/dd-20-28-209/avs2/src /home/training/dd-20-28-209/avs2/src /home/training/dd-20-28-209/avs2/build /home/training/dd-20-28-209/avs2/build /home/training/dd-20-28-209/avs2/build/CMakeFiles/PMC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PMC.dir/depend
