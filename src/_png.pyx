from libc cimport stdio, string
from cpython cimport exc, object
from cpython.bytes cimport PyBytes_AsStringAndSize, PyBytes_FromStringAndSize
cimport numpy as np

# Must do this, or the module will segfault
np.import_array()

import sys
# import collections


# How to get the message out?
cdef void raise_py_error():
    raise RuntimeError()


## DECLARATIONS


cdef extern from "stdio.h":
    stdio.FILE *fdopen(int fd, char *mode)


cdef extern from "Python.h":
    int PyFile_CheckExact(object)
    stdio.FILE* PyFile_AsFile(object)
    int PyObject_AsReadBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)
    bint PyObject_CheckReadBuffer(object obj)


cdef extern from "_png_core.h":
    ctypedef char png_byte
    ctypedef char* png_bytep
    ctypedef size_t png_size_t
    ctypedef size_t png_uint_32
    ctypedef void* png_voidp
    ctypedef void* png_structp
    ctypedef void (* png_rw_ptr)(png_structp, png_bytep, png_size_t)
    ctypedef void (* png_flush_ptr)(png_structp)

    void write_png_c "write_png" (
            png_byte* pix_buffer, png_uint_32 width,
            png_uint_32 height, stdio.FILE* file,
            png_voidp data_ptr, png_rw_ptr write_func,
            png_flush_ptr flush_func, double dpi) except +raise_py_error

    png_byte* read_png_c "read_png" (
            stdio.FILE* file, png_voidp data_ptr, png_rw_ptr read_func,
            int output_type, size_t* dimensions,
            size_t* bit_depth) except +raise_py_error


cdef extern from "png.h":
    void *png_get_io_ptr(png_structp png_ptr)


## END OF DECLARATIONS


cdef public void _write_png_callback(
    png_structp png_ptr, png_bytep data, png_size_t length):
    file_obj = <object>png_get_io_ptr(png_ptr)
    content = PyBytes_FromStringAndSize(<char *>data, length)
    file_obj.write(content)


cdef public void _flush_png_callback(png_structp png_ptr):
    file_obj = <object>png_get_io_ptr(png_ptr)
    file_obj.flush()


def write_png(buff, Py_ssize_t width, Py_ssize_t height, file_obj, double dpi=0.0):
    cdef void* pix_buffer
    cdef Py_ssize_t pix_buffer_length
    cdef int fd
    cdef stdio.FILE* fp

    if not PyObject_CheckReadBuffer(buff):
        raise TypeError("First argument must be a 32-bit RGBA buffer")

    if PyObject_AsReadBuffer(buff, &pix_buffer, &pix_buffer_length):
        raise ValueError("Couldn't get data from buffer")

    if pix_buffer_length != (width * height * 4):
        raise ValueError("Buffer size doesn't match given width and height")

    if sys.version_info[0] >= 3:
        fd = object.PyObject_AsFileDescriptor(file_obj)
        exc.PyErr_Clear()
        if fd != -1:
            fp = fdopen(fd, "w");
            write_png_c(
                <png_byte*>pix_buffer, width, height, fp,
                NULL, NULL, NULL, dpi)
            return
    else:
        if PyFile_CheckExact(file_obj):
            fp = PyFile_AsFile(file_obj)
            write_png_c(
                <png_byte*>pix_buffer, width, height, fp,
                NULL, NULL, NULL, dpi)
            return

    # if not isinstance(file_obj.write, collections.Callable):
    #     raise TypeError("output object is not a Python file-like object")

    write_png_c(
        <png_byte*>pix_buffer, width, height, NULL,
        <void *>file_obj, _write_png_callback, _flush_png_callback, dpi)


cdef public void _read_png_callback(
    png_structp png_ptr, png_bytep data, png_size_t length):
    cdef char* buff
    cdef Py_ssize_t bufflen

    file_obj = <object>png_get_io_ptr(png_ptr)
    content = file_obj.read(length)
    if PyBytes_AsStringAndSize(content, &buff, &bufflen) == 0:
        if bufflen == <Py_ssize_t>length:
            string.memcpy(data, buff, length)


def _read_png(file_obj, output_type):
    cdef int fd
    cdef stdio.FILE *fp
    cdef size_t dimensions[3]
    cdef size_t bit_depth
    cdef bint read_file = False
    cdef png_byte *buff

    if sys.version_info[0] >= 3:
        fd = object.PyObject_AsFileDescriptor(file_obj)
        exc.PyErr_Clear()
        if (fd != -1):
            fp = fdopen(fd, "r")
            buff = read_png_c(
                fp, NULL, NULL, output_type, dimensions, &bit_depth)
            read_file = True
    else:
        if PyFile_CheckExact(file_obj):
            fp = PyFile_AsFile(file_obj)
            buff = read_png_c(
                fp, NULL, NULL, output_type, dimensions, &bit_depth)
            read_file = True

    if not read_file:
        # if not isinstance(file_obj.read, collections.Callable):
        #     raise TypeError("input object is not a Python file-like object")

        buff = read_png_c(
            NULL, <void *>file_obj, _read_png_callback, output_type,
            dimensions, &bit_depth)

    cdef np.npy_intp numpy_dimensions[3]
    for i in range(3):
        numpy_dimensions[i] = <np.npy_intp>dimensions[i]

    cdef np.npy_intp ndim
    if numpy_dimensions[2] == 1:
        ndim = 2
    else:
        ndim = 3

    cdef int type
    if output_type == 1:
        type = np.NPY_FLOAT
    else:
        if bit_depth == 8:
            type = np.NPY_UINT8
        else:
            type = np.NPY_UINT16

    result = np.PyArray_SimpleNewFromData(ndim, numpy_dimensions, type, buff)

    return result


def read_png_float(file_obj):
    return _read_png(file_obj, 1)


def read_png_int(file_obj):
    return _read_png(file_obj, 0)


def read_png(file_obj):
    return _read_png(file_obj, 1)
