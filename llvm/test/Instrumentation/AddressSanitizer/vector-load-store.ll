; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=100 -S \
; RUN:     | FileCheck %s
; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -S \
; RUN:     | FileCheck %s -check-prefix=CALLS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @load.v1i32(ptr %p) sanitize_address {
; CHECK-LABEL: @load.v1i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = or i64 [[TMP2]], 17592186044416
; CHECK-NEXT:    [[TMP4:%.*]] = inttoptr i64 [[TMP3]] to ptr
; CHECK-NEXT:    [[TMP5:%.*]] = load i8, ptr [[TMP4]], align 1
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i8 [[TMP5]], 0
; CHECK-NEXT:    br i1 [[TMP6]], label [[TMP7:%.*]], label [[TMP13:%.*]], !prof [[PROF0:![0-9]+]]
; CHECK:       7:
; CHECK-NEXT:    [[TMP8:%.*]] = and i64 [[TMP1]], 7
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[TMP8]], 3
; CHECK-NEXT:    [[TMP10:%.*]] = trunc i64 [[TMP9]] to i8
; CHECK-NEXT:    [[TMP11:%.*]] = icmp sge i8 [[TMP10]], [[TMP5]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[TMP12:%.*]], label [[TMP13]]
; CHECK:       12:
; CHECK-NEXT:    call void @__asan_report_load4(i64 [[TMP1]]) #[[ATTR3:[0-9]+]]
; CHECK-NEXT:    unreachable
; CHECK:       13:
; CHECK-NEXT:    [[TMP14:%.*]] = load <1 x i32>, ptr [[P]], align 4
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @load.v1i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_load4(i64 [[TMP1]])
; CALLS-NEXT:    [[TMP2:%.*]] = load <1 x i32>, ptr [[P]], align 4
; CALLS-NEXT:    ret void
;
  load <1 x i32>, ptr %p
  ret void
}

define void @load.v2i32(ptr %p) sanitize_address {
; CHECK-LABEL: @load.v2i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = or i64 [[TMP2]], 17592186044416
; CHECK-NEXT:    [[TMP4:%.*]] = inttoptr i64 [[TMP3]] to ptr
; CHECK-NEXT:    [[TMP5:%.*]] = load i8, ptr [[TMP4]], align 1
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i8 [[TMP5]], 0
; CHECK-NEXT:    br i1 [[TMP6]], label [[TMP7:%.*]], label [[TMP8:%.*]]
; CHECK:       7:
; CHECK-NEXT:    call void @__asan_report_load8(i64 [[TMP1]]) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       8:
; CHECK-NEXT:    [[TMP9:%.*]] = load <2 x i32>, ptr [[P]], align 8
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @load.v2i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_load8(i64 [[TMP1]])
; CALLS-NEXT:    [[TMP2:%.*]] = load <2 x i32>, ptr [[P]], align 8
; CALLS-NEXT:    ret void
;
  load <2 x i32>, ptr %p
  ret void
}

define void @load.v4i32(ptr %p) sanitize_address {
; CHECK-LABEL: @load.v4i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = or i64 [[TMP2]], 17592186044416
; CHECK-NEXT:    [[TMP4:%.*]] = inttoptr i64 [[TMP3]] to ptr
; CHECK-NEXT:    [[TMP5:%.*]] = load i16, ptr [[TMP4]], align 2
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i16 [[TMP5]], 0
; CHECK-NEXT:    br i1 [[TMP6]], label [[TMP7:%.*]], label [[TMP8:%.*]]
; CHECK:       7:
; CHECK-NEXT:    call void @__asan_report_load16(i64 [[TMP1]]) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       8:
; CHECK-NEXT:    [[TMP9:%.*]] = load <4 x i32>, ptr [[P]], align 16
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @load.v4i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_load16(i64 [[TMP1]])
; CALLS-NEXT:    [[TMP2:%.*]] = load <4 x i32>, ptr [[P]], align 16
; CALLS-NEXT:    ret void
;
  load <4 x i32>, ptr %p
  ret void
}

define void @load.v8i32(ptr %p) sanitize_address {
; CHECK-LABEL: @load.v8i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = add i64 [[TMP1]], 31
; CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
; CHECK-NEXT:    [[TMP4:%.*]] = ptrtoint ptr [[P]] to i64
; CHECK-NEXT:    [[TMP5:%.*]] = lshr i64 [[TMP4]], 3
; CHECK-NEXT:    [[TMP6:%.*]] = or i64 [[TMP5]], 17592186044416
; CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
; CHECK-NEXT:    [[TMP8:%.*]] = load i8, ptr [[TMP7]], align 1
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ne i8 [[TMP8]], 0
; CHECK-NEXT:    br i1 [[TMP9]], label [[TMP10:%.*]], label [[TMP15:%.*]], !prof [[PROF0]]
; CHECK:       10:
; CHECK-NEXT:    [[TMP11:%.*]] = and i64 [[TMP4]], 7
; CHECK-NEXT:    [[TMP12:%.*]] = trunc i64 [[TMP11]] to i8
; CHECK-NEXT:    [[TMP13:%.*]] = icmp sge i8 [[TMP12]], [[TMP8]]
; CHECK-NEXT:    br i1 [[TMP13]], label [[TMP14:%.*]], label [[TMP15]]
; CHECK:       14:
; CHECK-NEXT:    call void @__asan_report_load_n(i64 [[TMP4]], i64 32) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       15:
; CHECK-NEXT:    [[TMP16:%.*]] = ptrtoint ptr [[TMP3]] to i64
; CHECK-NEXT:    [[TMP17:%.*]] = lshr i64 [[TMP16]], 3
; CHECK-NEXT:    [[TMP18:%.*]] = or i64 [[TMP17]], 17592186044416
; CHECK-NEXT:    [[TMP19:%.*]] = inttoptr i64 [[TMP18]] to ptr
; CHECK-NEXT:    [[TMP20:%.*]] = load i8, ptr [[TMP19]], align 1
; CHECK-NEXT:    [[TMP21:%.*]] = icmp ne i8 [[TMP20]], 0
; CHECK-NEXT:    br i1 [[TMP21]], label [[TMP22:%.*]], label [[TMP27:%.*]], !prof [[PROF0]]
; CHECK:       22:
; CHECK-NEXT:    [[TMP23:%.*]] = and i64 [[TMP16]], 7
; CHECK-NEXT:    [[TMP24:%.*]] = trunc i64 [[TMP23]] to i8
; CHECK-NEXT:    [[TMP25:%.*]] = icmp sge i8 [[TMP24]], [[TMP20]]
; CHECK-NEXT:    br i1 [[TMP25]], label [[TMP26:%.*]], label [[TMP27]]
; CHECK:       26:
; CHECK-NEXT:    call void @__asan_report_load_n(i64 [[TMP16]], i64 32) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       27:
; CHECK-NEXT:    [[TMP28:%.*]] = load <8 x i32>, ptr [[P]], align 32
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @load.v8i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_loadN(i64 [[TMP1]], i64 32)
; CALLS-NEXT:    [[TMP2:%.*]] = load <8 x i32>, ptr [[P]], align 32
; CALLS-NEXT:    ret void
;
  load <8 x i32>, ptr %p
  ret void
}

define void @load.v16i32(ptr %p) sanitize_address {
; CHECK-LABEL: @load.v16i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = add i64 [[TMP1]], 63
; CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
; CHECK-NEXT:    [[TMP4:%.*]] = ptrtoint ptr [[P]] to i64
; CHECK-NEXT:    [[TMP5:%.*]] = lshr i64 [[TMP4]], 3
; CHECK-NEXT:    [[TMP6:%.*]] = or i64 [[TMP5]], 17592186044416
; CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
; CHECK-NEXT:    [[TMP8:%.*]] = load i8, ptr [[TMP7]], align 1
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ne i8 [[TMP8]], 0
; CHECK-NEXT:    br i1 [[TMP9]], label [[TMP10:%.*]], label [[TMP15:%.*]], !prof [[PROF0]]
; CHECK:       10:
; CHECK-NEXT:    [[TMP11:%.*]] = and i64 [[TMP4]], 7
; CHECK-NEXT:    [[TMP12:%.*]] = trunc i64 [[TMP11]] to i8
; CHECK-NEXT:    [[TMP13:%.*]] = icmp sge i8 [[TMP12]], [[TMP8]]
; CHECK-NEXT:    br i1 [[TMP13]], label [[TMP14:%.*]], label [[TMP15]]
; CHECK:       14:
; CHECK-NEXT:    call void @__asan_report_load_n(i64 [[TMP4]], i64 64) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       15:
; CHECK-NEXT:    [[TMP16:%.*]] = ptrtoint ptr [[TMP3]] to i64
; CHECK-NEXT:    [[TMP17:%.*]] = lshr i64 [[TMP16]], 3
; CHECK-NEXT:    [[TMP18:%.*]] = or i64 [[TMP17]], 17592186044416
; CHECK-NEXT:    [[TMP19:%.*]] = inttoptr i64 [[TMP18]] to ptr
; CHECK-NEXT:    [[TMP20:%.*]] = load i8, ptr [[TMP19]], align 1
; CHECK-NEXT:    [[TMP21:%.*]] = icmp ne i8 [[TMP20]], 0
; CHECK-NEXT:    br i1 [[TMP21]], label [[TMP22:%.*]], label [[TMP27:%.*]], !prof [[PROF0]]
; CHECK:       22:
; CHECK-NEXT:    [[TMP23:%.*]] = and i64 [[TMP16]], 7
; CHECK-NEXT:    [[TMP24:%.*]] = trunc i64 [[TMP23]] to i8
; CHECK-NEXT:    [[TMP25:%.*]] = icmp sge i8 [[TMP24]], [[TMP20]]
; CHECK-NEXT:    br i1 [[TMP25]], label [[TMP26:%.*]], label [[TMP27]]
; CHECK:       26:
; CHECK-NEXT:    call void @__asan_report_load_n(i64 [[TMP16]], i64 64) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       27:
; CHECK-NEXT:    [[TMP28:%.*]] = load <16 x i32>, ptr [[P]], align 64
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @load.v16i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_loadN(i64 [[TMP1]], i64 64)
; CALLS-NEXT:    [[TMP2:%.*]] = load <16 x i32>, ptr [[P]], align 64
; CALLS-NEXT:    ret void
;
  load <16 x i32>, ptr %p
  ret void
}


define void @store.v1i32(ptr %p) sanitize_address {
; CHECK-LABEL: @store.v1i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = or i64 [[TMP2]], 17592186044416
; CHECK-NEXT:    [[TMP4:%.*]] = inttoptr i64 [[TMP3]] to ptr
; CHECK-NEXT:    [[TMP5:%.*]] = load i8, ptr [[TMP4]], align 1
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i8 [[TMP5]], 0
; CHECK-NEXT:    br i1 [[TMP6]], label [[TMP7:%.*]], label [[TMP13:%.*]], !prof [[PROF0]]
; CHECK:       7:
; CHECK-NEXT:    [[TMP8:%.*]] = and i64 [[TMP1]], 7
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[TMP8]], 3
; CHECK-NEXT:    [[TMP10:%.*]] = trunc i64 [[TMP9]] to i8
; CHECK-NEXT:    [[TMP11:%.*]] = icmp sge i8 [[TMP10]], [[TMP5]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[TMP12:%.*]], label [[TMP13]]
; CHECK:       12:
; CHECK-NEXT:    call void @__asan_report_store4(i64 [[TMP1]]) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       13:
; CHECK-NEXT:    store <1 x i32> zeroinitializer, ptr [[P]], align 4
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @store.v1i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_store4(i64 [[TMP1]])
; CALLS-NEXT:    store <1 x i32> zeroinitializer, ptr [[P]], align 4
; CALLS-NEXT:    ret void
;
  store <1 x i32> zeroinitializer, ptr %p
  ret void
}

define void @store.v2i32(ptr %p) sanitize_address {
; CHECK-LABEL: @store.v2i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = or i64 [[TMP2]], 17592186044416
; CHECK-NEXT:    [[TMP4:%.*]] = inttoptr i64 [[TMP3]] to ptr
; CHECK-NEXT:    [[TMP5:%.*]] = load i8, ptr [[TMP4]], align 1
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i8 [[TMP5]], 0
; CHECK-NEXT:    br i1 [[TMP6]], label [[TMP7:%.*]], label [[TMP8:%.*]]
; CHECK:       7:
; CHECK-NEXT:    call void @__asan_report_store8(i64 [[TMP1]]) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       8:
; CHECK-NEXT:    store <2 x i32> zeroinitializer, ptr [[P]], align 8
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @store.v2i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_store8(i64 [[TMP1]])
; CALLS-NEXT:    store <2 x i32> zeroinitializer, ptr [[P]], align 8
; CALLS-NEXT:    ret void
;
  store <2 x i32> zeroinitializer, ptr %p
  ret void
}

define void @store.v4i32(ptr %p) sanitize_address {
; CHECK-LABEL: @store.v4i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = or i64 [[TMP2]], 17592186044416
; CHECK-NEXT:    [[TMP4:%.*]] = inttoptr i64 [[TMP3]] to ptr
; CHECK-NEXT:    [[TMP5:%.*]] = load i16, ptr [[TMP4]], align 2
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i16 [[TMP5]], 0
; CHECK-NEXT:    br i1 [[TMP6]], label [[TMP7:%.*]], label [[TMP8:%.*]]
; CHECK:       7:
; CHECK-NEXT:    call void @__asan_report_store16(i64 [[TMP1]]) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       8:
; CHECK-NEXT:    store <4 x i32> zeroinitializer, ptr [[P]], align 16
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @store.v4i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_store16(i64 [[TMP1]])
; CALLS-NEXT:    store <4 x i32> zeroinitializer, ptr [[P]], align 16
; CALLS-NEXT:    ret void
;
  store <4 x i32> zeroinitializer, ptr %p
  ret void
}

define void @store.v8i32(ptr %p) sanitize_address {
; CHECK-LABEL: @store.v8i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = add i64 [[TMP1]], 31
; CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
; CHECK-NEXT:    [[TMP4:%.*]] = ptrtoint ptr [[P]] to i64
; CHECK-NEXT:    [[TMP5:%.*]] = lshr i64 [[TMP4]], 3
; CHECK-NEXT:    [[TMP6:%.*]] = or i64 [[TMP5]], 17592186044416
; CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
; CHECK-NEXT:    [[TMP8:%.*]] = load i8, ptr [[TMP7]], align 1
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ne i8 [[TMP8]], 0
; CHECK-NEXT:    br i1 [[TMP9]], label [[TMP10:%.*]], label [[TMP15:%.*]], !prof [[PROF0]]
; CHECK:       10:
; CHECK-NEXT:    [[TMP11:%.*]] = and i64 [[TMP4]], 7
; CHECK-NEXT:    [[TMP12:%.*]] = trunc i64 [[TMP11]] to i8
; CHECK-NEXT:    [[TMP13:%.*]] = icmp sge i8 [[TMP12]], [[TMP8]]
; CHECK-NEXT:    br i1 [[TMP13]], label [[TMP14:%.*]], label [[TMP15]]
; CHECK:       14:
; CHECK-NEXT:    call void @__asan_report_store_n(i64 [[TMP4]], i64 32) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       15:
; CHECK-NEXT:    [[TMP16:%.*]] = ptrtoint ptr [[TMP3]] to i64
; CHECK-NEXT:    [[TMP17:%.*]] = lshr i64 [[TMP16]], 3
; CHECK-NEXT:    [[TMP18:%.*]] = or i64 [[TMP17]], 17592186044416
; CHECK-NEXT:    [[TMP19:%.*]] = inttoptr i64 [[TMP18]] to ptr
; CHECK-NEXT:    [[TMP20:%.*]] = load i8, ptr [[TMP19]], align 1
; CHECK-NEXT:    [[TMP21:%.*]] = icmp ne i8 [[TMP20]], 0
; CHECK-NEXT:    br i1 [[TMP21]], label [[TMP22:%.*]], label [[TMP27:%.*]], !prof [[PROF0]]
; CHECK:       22:
; CHECK-NEXT:    [[TMP23:%.*]] = and i64 [[TMP16]], 7
; CHECK-NEXT:    [[TMP24:%.*]] = trunc i64 [[TMP23]] to i8
; CHECK-NEXT:    [[TMP25:%.*]] = icmp sge i8 [[TMP24]], [[TMP20]]
; CHECK-NEXT:    br i1 [[TMP25]], label [[TMP26:%.*]], label [[TMP27]]
; CHECK:       26:
; CHECK-NEXT:    call void @__asan_report_store_n(i64 [[TMP16]], i64 32) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       27:
; CHECK-NEXT:    store <8 x i32> zeroinitializer, ptr [[P]], align 32
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @store.v8i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_storeN(i64 [[TMP1]], i64 32)
; CALLS-NEXT:    store <8 x i32> zeroinitializer, ptr [[P]], align 32
; CALLS-NEXT:    ret void
;
  store <8 x i32> zeroinitializer, ptr %p
  ret void
}

define void @store.v16i32(ptr %p) sanitize_address {
; CHECK-LABEL: @store.v16i32(
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = add i64 [[TMP1]], 63
; CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
; CHECK-NEXT:    [[TMP4:%.*]] = ptrtoint ptr [[P]] to i64
; CHECK-NEXT:    [[TMP5:%.*]] = lshr i64 [[TMP4]], 3
; CHECK-NEXT:    [[TMP6:%.*]] = or i64 [[TMP5]], 17592186044416
; CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
; CHECK-NEXT:    [[TMP8:%.*]] = load i8, ptr [[TMP7]], align 1
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ne i8 [[TMP8]], 0
; CHECK-NEXT:    br i1 [[TMP9]], label [[TMP10:%.*]], label [[TMP15:%.*]], !prof [[PROF0]]
; CHECK:       10:
; CHECK-NEXT:    [[TMP11:%.*]] = and i64 [[TMP4]], 7
; CHECK-NEXT:    [[TMP12:%.*]] = trunc i64 [[TMP11]] to i8
; CHECK-NEXT:    [[TMP13:%.*]] = icmp sge i8 [[TMP12]], [[TMP8]]
; CHECK-NEXT:    br i1 [[TMP13]], label [[TMP14:%.*]], label [[TMP15]]
; CHECK:       14:
; CHECK-NEXT:    call void @__asan_report_store_n(i64 [[TMP4]], i64 64) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       15:
; CHECK-NEXT:    [[TMP16:%.*]] = ptrtoint ptr [[TMP3]] to i64
; CHECK-NEXT:    [[TMP17:%.*]] = lshr i64 [[TMP16]], 3
; CHECK-NEXT:    [[TMP18:%.*]] = or i64 [[TMP17]], 17592186044416
; CHECK-NEXT:    [[TMP19:%.*]] = inttoptr i64 [[TMP18]] to ptr
; CHECK-NEXT:    [[TMP20:%.*]] = load i8, ptr [[TMP19]], align 1
; CHECK-NEXT:    [[TMP21:%.*]] = icmp ne i8 [[TMP20]], 0
; CHECK-NEXT:    br i1 [[TMP21]], label [[TMP22:%.*]], label [[TMP27:%.*]], !prof [[PROF0]]
; CHECK:       22:
; CHECK-NEXT:    [[TMP23:%.*]] = and i64 [[TMP16]], 7
; CHECK-NEXT:    [[TMP24:%.*]] = trunc i64 [[TMP23]] to i8
; CHECK-NEXT:    [[TMP25:%.*]] = icmp sge i8 [[TMP24]], [[TMP20]]
; CHECK-NEXT:    br i1 [[TMP25]], label [[TMP26:%.*]], label [[TMP27]]
; CHECK:       26:
; CHECK-NEXT:    call void @__asan_report_store_n(i64 [[TMP16]], i64 64) #[[ATTR3]]
; CHECK-NEXT:    unreachable
; CHECK:       27:
; CHECK-NEXT:    store <16 x i32> zeroinitializer, ptr [[P]], align 64
; CHECK-NEXT:    ret void
;
; CALLS-LABEL: @store.v16i32(
; CALLS-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CALLS-NEXT:    call void @__asan_storeN(i64 [[TMP1]], i64 64)
; CALLS-NEXT:    store <16 x i32> zeroinitializer, ptr [[P]], align 64
; CALLS-NEXT:    ret void
;
  store <16 x i32> zeroinitializer, ptr %p
  ret void
}
