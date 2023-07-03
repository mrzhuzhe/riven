project(CPPTest)

# cmake -S src -B build
# cmake --build build

add_executable(2darray.o 2darray.cpp)

add_executable(boxcalculate.o boxcalculate.cpp)

add_executable(copylib.o copylib.cpp)
set_property(TARGET copylib.o PROPERTY CXX_STANDARD 17)


add_executable(functiontest.o functiontest.cpp)
set_property(TARGET functiontest.o PROPERTY CXX_STANDARD 17)

add_executable(iteratortest.o iteratortest.cpp)
set_property(TARGET iteratortest.o PROPERTY CXX_STANDARD 17)

add_executable(structtest.o structtest.cpp)

add_executable(template.o template.cpp)

add_executable(vectortest.o vectortest.cpp)
set_property(TARGET vectortest.o PROPERTY CXX_STANDARD 17)

add_executable(sharedpointer.o sharedpointer.cpp)
set_property(TARGET sharedpointer.o PROPERTY CXX_STANDARD 17)

add_executable(templateTest.o templateTest.cpp)
set_property(TARGET templateTest.o PROPERTY CXX_STANDARD 17)

# lib test
#add_library(sales_data sales_data.h)

#add_executable(vectortest_withlib.o vectortest.cpp )

#target_link_libraries(vectortest_withlib.o sales_data)


#add_executable(vectortest_withlib2.o sales_data.cpp vectortest.cpp)


add_executable(vfn_test.o vfn_test.cpp)