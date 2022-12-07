project(CPPTest)

# cmake -S src -B build
# cmake --build build

add_executable(2darray.o 2darray.cpp)

add_executable(boxcalculate.o boxcalculate.cpp)

add_executable(copylib.o copylib.cpp)

add_executable(functiontest.o functiontest.cpp)

add_executable(iteratortest.o iteratortest.cpp)

add_executable(structtest.o structtest.cpp)

add_executable(template.o template.cpp)

add_executable(vectortest.o vectortest.cpp)

add_executable(sharedpointer.o sharedpointer.cpp)

add_executable(templateTest.o templateTest.cpp)

# lib test
#add_library(sales_data sales_data.h)

#add_executable(vectortest_withlib.o vectortest.cpp )

#target_link_libraries(vectortest_withlib.o sales_data)


#add_executable(vectortest_withlib2.o sales_data.cpp vectortest.cpp)