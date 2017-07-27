TEMPLATE = app
TARGET = voc_train

CONFIG += console
CONFIG += c++11
CONFIG -= app_bundle

QT += core xml

HEADERS += \
    vocgputrain.h \
    vocpredict.h \
    metaconfig.h \
    annotationreader.h

SOURCES += \
    main.cpp \
    vocgputrain.cpp \
    vocpredict.cpp \
    annotationreader.cpp \
    metaconfig.cpp

win32{
    INCLUDEPATH += $$OPENCV_DIR/include
    LIBS += -L$$OPENCV_DIR/x64/vc14/lib

    QMAKE_CXXFLAGS += /openmp

    CONFIG(debug, debug | release){
        LIBS += -lopencv_core320d -lopencv_highgui320d -lopencv_imgproc320d -lopencv_imgcodecs320d -lopencv_videoio320d
    }else{
        LIBS += -lopencv_core320 -lopencv_highgui320 -lopencv_imgproc320 -lopencv_imgcodecs320 -lopencv_videoio320
    }
}else{
    QMAKE_CXXFLAGS += -fopenmp
    LIBS += -L/usr/local/lib
    LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lgomp
}

CONFIG(debug, debug | release){
    DST = debug
}else{
    DST = release
}

OBJECTS_DIR = tmp/$$DST/obj
UI_DIR = tmp/$$DST/ui
MOC_DIR = tmp/$$DST/moc
RCC_DIR = tmp/$$DST/rcc

include(ml_algorithms/ct/ct.pri)
include(ml_algorithms/gpu/gpu.pri)
