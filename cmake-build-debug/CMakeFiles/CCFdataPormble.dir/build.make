# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.15

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\Software_installation\CLion 2019.3.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\Software_installation\CLion 2019.3.1\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/CCFdataPormble.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CCFdataPormble.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CCFdataPormble.dir/flags.make

CMakeFiles/CCFdataPormble.dir/Main.cpp.obj: CMakeFiles/CCFdataPormble.dir/flags.make
CMakeFiles/CCFdataPormble.dir/Main.cpp.obj: ../Main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CCFdataPormble.dir/Main.cpp.obj"
	D:\Software_installation\Dev-Cpp\MinGW64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\CCFdataPormble.dir\Main.cpp.obj -c E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\Main.cpp

CMakeFiles/CCFdataPormble.dir/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CCFdataPormble.dir/Main.cpp.i"
	D:\Software_installation\Dev-Cpp\MinGW64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\Main.cpp > CMakeFiles\CCFdataPormble.dir\Main.cpp.i

CMakeFiles/CCFdataPormble.dir/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CCFdataPormble.dir/Main.cpp.s"
	D:\Software_installation\Dev-Cpp\MinGW64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\Main.cpp -o CMakeFiles\CCFdataPormble.dir\Main.cpp.s

# Object files for target CCFdataPormble
CCFdataPormble_OBJECTS = \
"CMakeFiles/CCFdataPormble.dir/Main.cpp.obj"

# External object files for target CCFdataPormble
CCFdataPormble_EXTERNAL_OBJECTS =

CCFdataPormble.exe: CMakeFiles/CCFdataPormble.dir/Main.cpp.obj
CCFdataPormble.exe: CMakeFiles/CCFdataPormble.dir/build.make
CCFdataPormble.exe: CMakeFiles/CCFdataPormble.dir/linklibs.rsp
CCFdataPormble.exe: CMakeFiles/CCFdataPormble.dir/objects1.rsp
CCFdataPormble.exe: CMakeFiles/CCFdataPormble.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CCFdataPormble.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\CCFdataPormble.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CCFdataPormble.dir/build: CCFdataPormble.exe

.PHONY : CMakeFiles/CCFdataPormble.dir/build

CMakeFiles/CCFdataPormble.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\CCFdataPormble.dir\cmake_clean.cmake
.PHONY : CMakeFiles/CCFdataPormble.dir/clean

CMakeFiles/CCFdataPormble.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020 E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020 E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\cmake-build-debug E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\cmake-build-debug E:\Workbench\CorCppFile\JBCL\HWSoftwareChallenge2020\cmake-build-debug\CMakeFiles\CCFdataPormble.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CCFdataPormble.dir/depend

