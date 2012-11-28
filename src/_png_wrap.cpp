/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
  Python wrappers for PNG reading and writing.
*/

#include "Python.h"
#include "mplutils.h"
#include "numpy/arrayobject.h"
#include "_png.h"


static void write_png_callback(png_structp png_ptr, png_bytep data,
                               png_size_t length)
{
    PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
    PyObject* write_method = PyObject_GetAttrString(py_file_obj, "write");
    PyObject* result = NULL;
    if (write_method) {
#if PY3K
        result = PyObject_CallFunction(write_method, (char *)"y#", data,
                                       length);
#else
        result = PyObject_CallFunction(write_method, (char *)"s#", data,
                                       length);
#endif
    }
    Py_XDECREF(write_method);
    Py_XDECREF(result);
}


static void flush_png_callback(png_structp png_ptr)
{
    PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
    PyObject* flush_method = PyObject_GetAttrString(py_file_obj, "flush");
    PyObject* result = NULL;
    if (flush_method) {
        result = PyObject_CallFunction(flush_method, (char *)"");
    }
    Py_XDECREF(flush_method);
    Py_XDECREF(result);
}


static PyObject* write_png_wrapper(PyObject* self, PyObject* args)
{
    PyObject* buffer = NULL;
    Py_ssize_t width = 0;
    Py_ssize_t height = 0;
    PyObject* file_obj = NULL;
    double dpi = 0.0;

    if (!PyArg_ParseTuple(
            args, "OnnO|f:write_png",
            &buffer, &width, &height, &file_obj, &dpi)) {
        return NULL;
    }

    if (!PyObject_CheckReadBuffer(buffer)) {
        PyErr_SetString(
            PyExc_TypeError, "First argument must be a 32-bit rgba buffer");
        return NULL;
    }

    const void* pix_buffer = NULL;
    Py_ssize_t pix_buffer_length = 0;
    if (PyObject_AsReadBuffer(buffer, &pix_buffer, &pix_buffer_length)) {
        PyErr_SetString(
            PyExc_ValueError, "Couldn't get data from read buffer");
        return NULL;
    }

    if (pix_buffer_length < width * height * 4) {
        PyErr_SetString(
            PyExc_ValueError,
            "Buffer size doesn't match given width and height");
        return NULL;
    }

#if PY3K
    int fd = PyObject_AsFileDescriptor(file_obj);
    PyErr_Clear();
    if (fd != -1) {
        FILE* fp = fdopen(fd, "w");
#else // not PY3K
    if (PyFile_CheckExact(file_obj)) {
        FILE* fp = PyFile_AsFile(file_obj);
#endif
        try {
            write_png((png_byte*)pix_buffer, width, height, fp,
                NULL, NULL, NULL, dpi);
        } catch (const char *e) {
            PyErr_SetString(PyExc_ValueError, e);
            return NULL;
        }
    } else {
        PyObject *write_method = PyObject_GetAttrString(file_obj, "write");
        if (!(write_method && PyCallable_Check(write_method))) {
            Py_XDECREF(write_method);
            PyErr_SetString(PyExc_TypeError,
                "Output object is not a Python file-like object");
            return NULL;
        }

        Py_DECREF(write_method);

        try {
            write_png((png_byte*)pix_buffer, width, height, NULL, file_obj,
                      write_png_callback, flush_png_callback, dpi);
        } catch (const char *e) {
            PyErr_SetString(PyExc_RuntimeError, e);
            return NULL;
        }
    }

    if (PyErr_Occurred()) {
        return NULL;
    } else {
        Py_RETURN_NONE;
    }
}


static void read_png_callback(png_structp png_ptr, png_bytep data, png_size_t length)
{
    PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
    PyObject* read_method = PyObject_GetAttrString(py_file_obj, "read");
    PyObject* result = NULL;
    char *buffer;
    Py_ssize_t bufflen;
    if (read_method) {
        result = PyObject_CallFunction(read_method, (char *)"i", length);
        if (PyBytes_AsStringAndSize(result, &buffer, &bufflen) == 0) {
            if (bufflen == (Py_ssize_t)length) {
                memcpy(data, buffer, length);
            }
        }
    }
    Py_XDECREF(read_method);
    Py_XDECREF(result);
}


static PyObject* _read_png_wrapper(
    PyObject* self, PyObject* args, output_buffer_t output_type)
{
    PyObject* fileobj;
    FILE* fp = NULL;
    png_byte* buffer = NULL;
    size_t dimensions[3];
    size_t bit_depth;

    if (!PyArg_ParseTuple(
            args, "O:read_png", &fileobj)) {
        return NULL;
    }

#if PY3K
    int fd = PyObject_AsFileDescriptor(fileobj);
    PyErr_Clear();
    if (fd != -1) {
        fp = fdopen(fd, "r");
#else
    if (PyFile_CheckExact(fileobj)) {
        fp = PyFile_AsFile(fileobj);
#endif
        try {
            buffer = read_png(fp, NULL, NULL, output_type,
                              dimensions, &bit_depth);
        } catch (const char *e) {
            PyErr_SetString(PyExc_RuntimeError, e);
            return NULL;
        }
    } else {
        PyObject* read_method = PyObject_GetAttrString(fileobj, "read");
        if (!(read_method && PyCallable_Check(read_method)))
        {
            Py_XDECREF(read_method);
            PyErr_SetString(
                PyExc_TypeError,
                "First argument is not a Python file-like object");
            return NULL;
        }

        Py_DECREF(read_method);

        try {
            buffer = read_png(NULL, fileobj, read_png_callback,
                              output_type, dimensions, &bit_depth);
        } catch (const char *e) {
            PyErr_SetString(PyExc_RuntimeError, e);
        }
    }

    if (PyErr_Occurred()) {
        return NULL;
    }

    npy_intp numpy_dimensions[3];
    for (size_t i = 0; i < 3; ++i) {
        numpy_dimensions[i] = (npy_intp)dimensions[i];
    }
    npy_intp ndim = (numpy_dimensions[2] == 1) ? 2 : 3;
    int type;
    if (output_type == OUTPUT_FLOAT) {
        type = NPY_FLOAT;
    } else {
        if (bit_depth == 8) {
            type = NPY_UINT8;
        } else {
            type = NPY_UINT16;
        }
    }

    PyObject* result = (PyObject *)PyArray_SimpleNewFromData(
        ndim, numpy_dimensions, type, buffer);
    if (result == NULL) {
        delete buffer;
        return NULL;
    }
    return result;
}


static PyObject* read_png_float_wrapper(PyObject* self, PyObject* args)
{
    return _read_png_wrapper(self, args, OUTPUT_FLOAT);
}


static PyObject* read_png_int_wrapper(PyObject* self, PyObject* args)
{
    return _read_png_wrapper(self, args, OUTPUT_UINT);
}


static PyMethodDef png_methods[] = {
    {"write_png", (PyCFunction)write_png_wrapper, METH_VARARGS,
     "write_png(buffer, width, height, fileobj, dpi=None)"},
    {"read_png", (PyCFunction)read_png_float_wrapper, METH_VARARGS,
     "read_png(fileobj)"},
    {"read_png_float", (PyCFunction)read_png_float_wrapper, METH_VARARGS,
     "read_png_float(fileobj)"},
    {"read_png_int", (PyCFunction)read_png_int_wrapper, METH_VARARGS,
     "read_png_int(fileobj)"}
};


extern "C" {
PyMODINIT_FUNC
#if PY3K
PyInit__png(void)
#else
init_png(void)
#endif
{
    import_array();

    PyObject *m;

    m = Py_InitModule3(
        "matplotlib._png", png_methods,
            "Utilities for reading/writing PNG files");

#if PY3K
    return m
#endif
}
}
