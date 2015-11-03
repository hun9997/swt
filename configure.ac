#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([swt], [0.1], [pwendt@gracenote.com])
AC_CONFIG_SRCDIR([include/swt.hpp])
AC_CONFIG_AUX_DIR([.])
AC_CONFIG_MACRO_DIR([m4])
#AM_INIT_AUTOMAKE(2.69, swt)
AM_INIT_AUTOMAKE([subdir-objects])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PATH_TOOL([PKG_CONFIG], [pkg-config])
AC_LANG([C++])
AX_CXX_COMPILE_STDCXX_11()
LT_INIT

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([malloc.h stdlib.h string.h sys/time.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_CHECK_FUNCS([floor gettimeofday memset sqrt])

AC_SUBST([CFLAGS])
AC_SUBST([LIBS])
AC_SUBST([AM_CFLAGS])
AC_SUBST([AM_LDFLAGS])

AC_CONFIG_FILES([Makefile src/Makefile include/Makefile test/Makefile])
AC_OUTPUT
