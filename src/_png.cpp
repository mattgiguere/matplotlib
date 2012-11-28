/* -*- mode: c++; c-basic-offset: 4 -*- */

#include "_png.h"

// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS

void write_png(png_byte* pix_buffer, const png_uint_32 width,
               const png_uint_32 height, FILE* file,
               const png_voidp data_ptr, const png_rw_ptr write_func,
               const png_flush_ptr flush_func, const double dpi)
    throw (const char *)
{
    png_bytep *row_pointers = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;

    struct png_color_8_struct sig_bit;
    png_uint_32 row;

    row_pointers = new png_bytep[height];
    for (row = 0; row < height; ++row) {
        row_pointers[row] = pix_buffer + row * width * 4;
    }

    try {
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (png_ptr == NULL) {
            throw "Could not create write struct";
        }

        info_ptr = png_create_info_struct(png_ptr);
        if (info_ptr == NULL) {
            throw "Could not create info struct";
        }

        if (setjmp(png_jmpbuf(png_ptr))) {
            throw "Error building image";
        }

        if (file) {
            png_init_io(png_ptr, file);
        } else {
            png_set_write_fn(png_ptr, data_ptr,
                             write_func, flush_func);
        }
        png_set_IHDR(png_ptr, info_ptr,
                     width, height, 8,
                     PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        // Save the dpi of the image in the file
        if (dpi != 0.0) {
            size_t dots_per_meter = (size_t)(dpi / (2.54 / 100.0));
            png_set_pHYs(png_ptr, info_ptr, dots_per_meter, dots_per_meter,
                         PNG_RESOLUTION_METER);
        }

        // this a a color image!
        sig_bit.gray = 0;
        sig_bit.red = 8;
        sig_bit.green = 8;
        sig_bit.blue = 8;
        /* if the image has an alpha channel then */
        sig_bit.alpha = 8;
        png_set_sBIT(png_ptr, info_ptr, &sig_bit);

        png_write_info(png_ptr, info_ptr);
        png_write_image(png_ptr, row_pointers);
        png_write_end(png_ptr, info_ptr);
    } catch (const char *e) {
        if (png_ptr && info_ptr) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
        }
        delete [] row_pointers;
        /* Changed calls to png_destroy_write_struct to follow
           http://www.libpng.org/pub/png/libpng-manual.txt.
           This ensures the info_ptr memory is released.
        */
        throw;
    }

    png_destroy_write_struct(&png_ptr, &info_ptr);
    delete [] row_pointers;
#if PY3K
    if (fp) {
        fflush(fp);
    }
#endif
}


png_byte* read_png(FILE* fp, png_voidp data_ptr, png_rw_ptr read_callback,
               output_buffer_t output_type, size_t* dimensions,
               size_t* bit_depth)
  throw (const char *)
{
    /* initialize stuff */
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr) {
        throw "png_create_read_struct failed";
    }

    png_byte header[8];   // 8 is the maximum size that can be checked
    if (fp) {
        if (fread(header, 1, 8, fp) != 8) {
            throw "Error reading PNG header";
        }
        png_init_io(png_ptr, fp);
    } else {
        png_set_read_fn(png_ptr, data_ptr, read_callback);
        read_callback(png_ptr, header, 8);
    }

    if (png_sig_cmp(header, 0, 8)) {
        throw "File not recognized as a PNG file";
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        throw "png_create_info_struct failed";
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        throw "error calling setjmp";
    }

    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    png_uint_32 width = png_get_image_width(png_ptr, info_ptr);
    png_uint_32 height = png_get_image_height(png_ptr, info_ptr);

    *bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Unpack 1, 2, and 4-bit images
    if (*bit_depth < 8) {
        png_set_packing(png_ptr);
        *bit_depth = 8;
    }

    // If sig bits are set, shift data
    png_color_8p sig_bit;
    if ((png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_PALETTE) &&
        png_get_sBIT(png_ptr, info_ptr, &sig_bit)) {
        png_set_shift(png_ptr, sig_bit);
    }

    // Convert big endian to little
    if (*bit_depth == 16) {
        png_set_swap(png_ptr);
    }

    // Convert palletes to full RGB
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
        *bit_depth = 8;
    }

    // If there's an alpha channel convert gray to RGB
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png_ptr);
    }

    png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    png_bytep *row_pointers = new png_bytep[height];
    png_uint_32 row;
    for (row = 0; row < height; row++) {
        row_pointers[row] = new png_byte[png_get_rowbytes(png_ptr, info_ptr)];
    }

    png_read_image(png_ptr, row_pointers);

    dimensions[0] = height;
    dimensions[1] = width;
    if (png_get_color_type(png_ptr, info_ptr) & PNG_COLOR_MASK_ALPHA) {
        dimensions[2] = 4;     //RGBA images
    } else if (png_get_color_type(png_ptr, info_ptr) & PNG_COLOR_MASK_COLOR) {
        dimensions[2] = 3;     //RGB images
    } else {
        dimensions[2] = 1;     //Greyscale images
    }

    size_t size = dimensions[0] * dimensions[1] * dimensions[2];
    png_byte *output_buffer = NULL;

    if (output_type == OUTPUT_FLOAT) {
        output_buffer = new png_byte[size * sizeof(float)];
        float *write_ptr = (float *)output_buffer;

        if (*bit_depth == 16) {
            for (png_uint_32 y = 0; y < height; ++y) {
                png_byte* row = row_pointers[y];
                png_uint_16* read_ptr = (png_uint_16*)row;
                for (png_uint_32 x = 0; x < width; ++x) {
                    for (png_uint_32 p = 0; p < dimensions[2];
                         ++p, ++write_ptr, ++read_ptr) {
                        *write_ptr++ = (float)(*read_ptr) / 65535.0;
                    }
                }
            }
        } else {
            for (png_uint_32 y = 0; y < height; ++y) {
                png_byte* row = row_pointers[y];
                png_byte* read_ptr = row;
                for (png_uint_32 x = 0; x < width; ++x) {
                    for (png_uint_32 p = 0; p < dimensions[2];
                         ++p, ++write_ptr, ++read_ptr) {
                        *write_ptr = (float)(*read_ptr) / 255.0;
                    }
                }
            }
        }
    } else {
        if (*bit_depth == 8) {
            output_buffer = new png_byte[size];
            png_byte* write_ptr = (png_byte *)output_buffer;

            for (png_uint_32 y = 0; y < height; ++y) {
                png_byte* row = row_pointers[y];
                png_byte* read_ptr = (png_byte*)row;
                for (png_uint_32 x = 0; x < width; ++x) {
                    for (png_uint_32 p = 0; p < dimensions[2];
                         ++p, ++write_ptr, ++read_ptr) {
                        *write_ptr = *read_ptr;
                    }
                }
            }
        } else {
            output_buffer = new png_byte[size * sizeof(png_uint_16)];
            png_uint_16* write_ptr = (png_uint_16 *)output_buffer;

            for (png_uint_32 y = 0; y < height; ++y) {
                png_byte* row = row_pointers[y];
                png_uint_16* read_ptr = (png_uint_16*)row;
                for (png_uint_32 x = 0; x < width; ++x) {
                    for (png_uint_32 p = 0; p < dimensions[2];
                         ++p, ++write_ptr, ++read_ptr) {
                        *write_ptr = *read_ptr;
                    }
                }
            }
        }
    }

    //free the png memory
    png_read_end(png_ptr, info_ptr);
#ifndef png_infopp_NULL
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
#else
    png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
#endif
    for (row = 0; row < height; row++) {
        delete[] row_pointers[row];
    }
    delete[] row_pointers;

    return output_buffer;
}
