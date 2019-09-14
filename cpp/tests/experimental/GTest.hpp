/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#pragma once

#ifdef GTEST_INCLUDE_GTEST_GTEST_H_
#error "Don't include gtest/gtest.h directly, include util/GTest.hpp instead"
#endif

/**---------------------------------------------------------------------------*
 * @file GTest.hpp
 * @brief Work around for GTests emulation of variadic templates in
 * ::Testing::Types.
 *
 * Removes the 50 type limit in a type-parameterized test list.
 *
 * Uses macros to rename GTests's emulated variadic template types and then
 * redefines them properly.
 *---------------------------------------------------------------------------**/

#define Types NV_VPI_Types_NOT_USED
#define Types0 NV_VPI_Types0_NOT_USED
#define TypeList NV_VPI_TypeList_NOT_USED
#define Templates NV_VPI_Templates_NOT_USED
#define Templates0 NV_VPI_Templates0_NOT_USED
#include <gtest/internal/gtest-type-util.h>
#undef Types
#undef Types0
#undef TypeList
#undef Templates
#undef Templates0

#include <cmath>  // for std::abs

namespace testing {

template <class... TYPES>
struct Types {
  using type = Types;
};

template <class T, class... TYPES>
struct Types<T, TYPES...> {
  using Head = T;
  using Tail = Types<TYPES...>;

  using type = Types;
};

namespace internal {

using Types0 = Types<>;

template <GTEST_TEMPLATE_... TYPES>
struct Templates {};

template <GTEST_TEMPLATE_ HEAD, GTEST_TEMPLATE_... TAIL>
struct Templates<HEAD, TAIL...> {
  using Head = internal::TemplateSel<HEAD>;
  using Tail = Templates<TAIL...>;

  using type = Templates;
};

using Templates0 = Templates<>;

template <typename T>
struct TypeList {
  typedef Types<T> type;
};

template <class... TYPES>
struct TypeList<Types<TYPES...>> {
  using type = Types<TYPES...>;
};

}  // namespace internal
}  // namespace testing

#include <gtest/gtest.h>
