/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef ___MATPLOTLIB_PNG_H__
#define ___MATPLOTLIB_PNG_H__

#include <png.h>

/*
Write to a PNG file.  This function is able to write to a regular C
FILE * or to provide callbacks to perform the writing.

Parameters
----------
pix_buffer : png_byte*
   The buffer containing the pixel data.  Must be 8-bit-per-plane RGBA
   and of size width x height.

width : png_uint_32
   The width of the image.

height : png_uint_32
   The height of the image.

file : FILE *
   An opened FILE pointer to write to.  Must be NULL if data_ptr is
   non-NULL.

data_ptr : png_voidp
   An arbitrary object that will be passed to the callback functions
   below.  Must be NULL if file is non-NULL.

write_func : png_rw_ptr
   A callback function to write bytes to an arbitrary file stream.

     static void (*)(png_structp png_ptr, png_bytep data,
                     png_size_t length)

   Within the callback, data_ptr can be obtained using
   ``png_get_io_ptr(png_ptr)``.

flush_func : png_flush_ptr
   A callback function to flush an arbitrary file stream.

     static void (*)(png_structp png_ptr)

   Within the callback, data_ptr can be obtained using
   ``png_get_io_ptr(png_ptr)``.

dpi : double
   The DPI of the image.  Used only to store the DPI in the PNG file's
   metadata.
*/
void write_png(png_byte* pix_buffer, const png_uint_32 width,
               const png_uint_32 height, FILE* file,
               const png_voidp data_ptr, const png_rw_ptr write_func,
               const png_flush_ptr flush_func, const double dpi)
    throw (const char *);


typedef enum output_buffer_t {
    OUTPUT_UINT,
    OUTPUT_FLOAT
} output_buffer_t;


/*
Read from a PNG file.  This function is able to read directly from a
regular C FILE * or use a callback function.

Parameters
----------
file : FILE *
   An opened FILE pointer to read from.  Must be NULL if data_ptr is
   non-NULL.

data_ptr : png_voidp
   An arbitrary object that will be passed to the callback function
   below.  Must be NULL if file is non-NULL.

read_func : png_rw_ptr
   A callback function to read bytes from an arbitrary file stream.

     static void (*)(png_structp png_ptr, png_bytep data,
                     png_size_t length)

   Within the callback, data_ptr can be obtained using
   ``png_get_io_ptr(png_ptr)``.

output_type : output_buffer_t
   The type of data to return.  May either OUTPUT_UINT or
   OUTPUT_FLOAT.

Returns
-------
dimensions : size_t[3]
   The dimensions of the output buffer, of the form {height, width,
   planes}.

bit_depth : size_t *
   If output_type is OUTPUT_UINT, returns the bit depth of the data,
   either 8 or 16.

return value : png_byte *
   The output buffer, allocated by this function.  It is the
   responsibility of the caller to free or otherwise take ownership of
   this memory.
 */
png_byte* read_png(FILE* file, png_voidp data_ptr, png_rw_ptr read_callback,
               output_buffer_t output_type, size_t *dimensions,
               size_t *bit_depth)
  throw (const char *);

#endif
