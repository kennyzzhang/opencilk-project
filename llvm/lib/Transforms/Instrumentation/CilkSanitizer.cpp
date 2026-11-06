//===- CilkSanitizer.cpp - Nondeterminism detector for Cilk/Tapir ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of CilkSan, a determinacy-race detector for Cilk
// programs.
//
// This instrumentation pass inserts calls to the runtime library before
// appropriate memory accesses.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/CilkSanitizer.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirRaceDetect.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Instrumentation/CSI.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cilksan"

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumOmittedReadsFromConstants, "Number of reads from constant data");
STATISTIC(NumOmittedNonCaptured, "Number of accesses ignored due to capturing");
STATISTIC(NumInstrumentedMemIntrinsicReads,
          "Number of instrumented reads from memory intrinsics");
STATISTIC(NumInstrumentedMemIntrinsicWrites,
          "Number of instrumented writes from memory intrinsics");
STATISTIC(NumInstrumentedDetaches, "Number of instrumented detaches");
STATISTIC(NumInstrumentedDetachExits, "Number of instrumented detach exits");
STATISTIC(NumInstrumentedSyncs, "Number of instrumented syncs");
STATISTIC(NumInstrumentedAllocas, "Number of instrumented allocas");
STATISTIC(NumInstrumentedAllocFns,
          "Number of instrumented allocation functions");
STATISTIC(NumInstrumentedFrees, "Number of instrumented free calls");
STATISTIC(
    NumHoistedInstrumentedReads,
    "Number of reads whose instrumentation has been coalesced and hoisted");
STATISTIC(
    NumHoistedInstrumentedWrites,
    "Number of writes whose instrumentation has been coalesced and hoisted");
STATISTIC(NumNestAccesses,
          "Number of accesses covered by checks hoisted before a loop nest");
STATISTIC(NumNestChecks, "Number of range checks hoisted before loop nests");
STATISTIC(NumSymNestAccesses, "Number of accesses covered by checks hoisted "
                              "before a loop nest with symbolic bounds");
STATISTIC(NumSymNestChecks, "Number of range-check sites (each possibly a "
                            "loop of checks) hoisted before symbolic nests");
STATISTIC(NumFootprintMarkers,
          "Number of footprint markers placed before symbolic nests");
STATISTIC(NumFootprintMarkersUsed,
          "Number of footprint markers turned into range checks");
STATISTIC(NumFootprintTagsPaired,
          "Number of tagged accesses paired with a footprint marker");
STATISTIC(NumFootprintTagsLost,
          "Number of tagged accesses without a footprint marker");
STATISTIC(NumBlockAccesses,
          "Number of accesses covered by range checks within a basic block");
STATISTIC(NumBlockChecks, "Number of range checks within basic blocks");
STATISTIC(NumMergedMAAPGuarded,
          "Number of range checks (nests, blocks) MAAP can skip");
STATISTIC(NumMergedMAAPLost,
          "Number of range checks (nests, blocks) MAAP could skip by their "
          "representative alone but not by every access they cover");
STATISTIC(NumSunkInstrumentedReads,
          "Number of reads whose instrumentation has been coalesced and sunk");
STATISTIC(NumSunkInstrumentedWrites,
          "Number of writes whose instrumentation has been coalesced and sunk");

static cl::opt<bool>
    EnableStaticRaceDetection(
        "enable-static-race-detection", cl::init(true), cl::Hidden,
        cl::desc("Enable static detection of determinacy races."));

static cl::opt<bool>
    AssumeRaceFreeLibraryFunctions(
        "assume-race-free-lib", cl::init(false), cl::Hidden,
        cl::desc("Assume library functions are race free."));

static cl::opt<bool>
    IgnoreInaccessibleMemory(
        "ignore-inaccessible-memory", cl::init(false), cl::Hidden,
        cl::desc("Ignore inaccessible memory when checking for races."));

static cl::opt<bool>
    AssumeNoExceptions(
        "cilksan-assume-no-exceptions", cl::init(false), cl::Hidden,
        cl::desc("Assume that ordinary calls cannot throw exceptions."));

static cl::opt<unsigned>
    MaxUsesToExploreCapture(
        "cilksan-max-uses-to-explore-capture", cl::init(unsigned(-1)),
        cl::Hidden,
        cl::desc("Maximum number of uses to explore for a capture query."));

static cl::opt<bool> MAAPChecks("cilksan-maap-checks", cl::init(true),
                                cl::Hidden,
                                cl::desc("Enable or disable MAAP checks."));

static cl::opt<bool> LoopHoisting(
    "cilksan-loop-hoisting", cl::init(true), cl::Hidden,
    cl::desc("Enable or disable hoisting instrumentation out of loops."));

static cl::opt<bool> HoistNests(
    "cilksan-hoist-nests", cl::init(false), cl::Hidden,
    cl::desc("Check the exact footprint of a serial loop nest with constant "
             "trip counts once, before the nest, instead of each access."));

static cl::opt<bool> HoistSymbolicNests(
    "cilksan-hoist-symbolic-nests", cl::init(false), cl::Hidden,
    cl::desc("Like -cilksan-hoist-nests, for nests whose trip counts and "
             "strides are fixed on entry but not constant: each access's "
             "footprint becomes one range check, or a loop of up to two "
             "levels of row checks, before the nest. The footprints are "
             "found before the vectorizer (see CilksanFootprintsPass)."));

static cl::opt<bool> CoalesceBlocks(
    "cilksan-coalesce-blocks", cl::init(false), cl::Hidden,
    cl::desc("Check the exact footprint of straight-line accesses at constant "
             "offsets from one base, between calls in a basic block, with "
             "range checks instead of each access (e.g. unrolled kernels)."));

// The instructions of S in the order they appear in their function. Iterating
// a SmallPtrSet follows pointer values, which differ from run to run, and
// that order decides the order of the code SCEVExpander emits.
static SmallVector<Instruction *, 8>
inProgramOrder(const SmallPtrSetImpl<Instruction *> &S) {
  SmallVector<Instruction *, 8> V(S.begin(), S.end());
  if (V.size() < 2)
    return V;
  DenseMap<const Instruction *, unsigned> Pos;
  unsigned N = 0;
  for (BasicBlock &BB : *V[0]->getFunction())
    for (Instruction &I : BB)
      if (S.count(&I))
        Pos[&I] = N++;
  llvm::sort(V, [&](Instruction *A, Instruction *B) { return Pos[A] < Pos[B]; });
  return V;
}

// The bytes a load or store accesses: its store size, unlike
// ScalarEvolution::getElementSize, which gives the type's allocation size and
// so rounds a vector like <6 x i64> (48 bytes) up to 64 on x86, making a range
// check claim bytes the access doesn't touch -- a false race with whoever
// writes them. Null for other instructions, like getElementSize.
static const SCEV *accessSize(ScalarEvolution &SE, Instruction *I) {
  Type *Ty;
  if (auto *St = dyn_cast<StoreInst>(I))
    Ty = St->getValueOperand()->getType();
  else if (auto *Ld = dyn_cast<LoadInst>(I))
    Ty = Ld->getType();
  else
    return nullptr;
  return SE.getStoreSizeOfExpr(
      SE.getEffectiveSCEVType(PointerType::getUnqual(I->getContext())), Ty);
}

// The size of a plain load or store (not volatile, unordered) with a
// constant size. None for anything else.
static std::optional<int64_t> plainAccessSize(Instruction &I,
                                              ScalarEvolution &SE) {
  if (auto *Ld = dyn_cast<LoadInst>(&I)) {
    if (Ld->isVolatile() || !Ld->isUnordered())
      return std::nullopt;
  } else if (auto *St = dyn_cast<StoreInst>(&I)) {
    if (St->isVolatile() || !St->isUnordered())
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  const auto *Sz = dyn_cast<SCEVConstant>(accessSize(SE, &I));
  if (!Sz)
    return std::nullopt;
  return Sz->getAPInt().getSExtValue();
}

// The size of an access the hoisting passes (HoistNests, HoistSymbolicNests,
// CoalesceBlocks) may cover: a plain access (see plainAccessSize) that would
// be checked, and not opaquely. None for anything else.
static std::optional<int64_t> hoistableAccessSize(Instruction &I, RaceInfo &RI,
                                                  ScalarEvolution &SE) {
  std::optional<int64_t> Size = plainAccessSize(I, SE);
  if (!Size || !(RI.mightRaceViaAncestor(&I) || RI.mightRaceLocally(&I)) ||
      RI.mightRaceOpaquely(&I))
    return std::nullopt;
  return Size;
}

// Footprint markers (see CilksanFootprintsPass): the function they call, the
// metadata kind that ties an access to its marker, and the attribute of a
// function the early pass has seen.
static constexpr char FootprintMarkerName[] = "__cilksan_fp";
static constexpr char FootprintTagKind[] = "cilksan.fp";
static constexpr char FootprintsAttr[] = "cilksan-footprints";
static bool checkSanitizeCilkAttr(Function &F);

// B - A, if it is a constant of at most 32 significant bits: the offset
// between two addresses from one base.
static std::optional<int64_t> constOffset(ScalarEvolution &SE, const SCEV *B,
                                          const SCEV *A) {
  const auto *D = dyn_cast<SCEVConstant>(SE.getMinusSCEV(B, A));
  if (!D || D->getAPInt().getSignificantBits() > 32)
    return std::nullopt;
  return D->getAPInt().getSExtValue();
}

// The checks for a footprint given as a map of the bytes read (1) and written
// (2): maximal runs of written bytes, and of bytes only read, as (start,
// length, written).
using ByteRun = std::tuple<int64_t, int64_t, bool>;
static SmallVector<ByteRun, 8> byteMapRuns(ArrayRef<uint8_t> Map) {
  SmallVector<ByteRun, 8> Runs;
  for (int64_t b = 0, n = Map.size(); b < n;) {
    if (!Map[b]) {
      ++b;
      continue;
    }
    bool W = Map[b] & 2;
    int64_t e = b;
    while (e < n && Map[e] && bool(Map[e] & 2) == W)
      ++e;
    Runs.emplace_back(b, e - b, W);
    b = e;
  }
  return Runs;
}

static cl::opt<bool>
    IgnoreSanitizeCilkAttr(
        "ignore-sanitize-cilk-attr", cl::init(false), cl::Hidden,
        cl::desc("Ignore the 'sanitize_cilk' attribute when choosing what to "
                 "instrument."));

static cl::opt<std::string> ClCilksanBCPath(
    "cilksan-bc-path", cl::init(""), cl::Hidden,
    cl::desc("Path to the bitcode file for the Cilksan library."));

static const unsigned SERIESPARALLEL = 0x1;
static const unsigned SHADOWMEMORY = 0x2;
static cl::opt<unsigned> InstrumentationSet(
    "cilksan-instrumentation-set", cl::init(SERIESPARALLEL | SHADOWMEMORY),
    cl::Hidden,
    cl::desc("Specify the set of instrumentation hooks to insert."));

static const char *const CsanRtUnitInitName = "__csanrt_unit_init";
static const char *const CsiUnitObjTableName = "__csi_unit_obj_table";
static const char *const CsiUnitObjTableArrayName = "__csi_unit_obj_tables";

/// Maintains a mapping from CSI ID of a load or store to the source information
/// of the object accessed by that load or store.
class ObjectTable : public ForensicTable {
public:
  ObjectTable() : ForensicTable() {}
  ObjectTable(Module &M, StringRef BaseIdName) : ForensicTable(M, BaseIdName) {}

  /// The number of entries in this table
  uint64_t size() const { return LocalIdToSourceLocationMap.size(); }

  /// Add the given instruction to this table.
  /// \returns The local ID of the Instruction.
  uint64_t add(Instruction &I, Value *Obj);

  /// Get the Type for a pointer to a table entry.
  ///
  /// A table entry is just a source location.
  static PointerType *getPointerType(LLVMContext &C);

  /// Insert this table into the given Module.
  ///
  /// The table is constructed as a ConstantArray indexed by local IDs.  The
  /// runtime is responsible for performing the mapping that allows the table to
  /// be indexed by global ID.
  Constant *insertIntoModule(Module &M) const;

private:
  struct SourceLocation {
    StringRef Name;
    int32_t Line;
    StringRef Filename;
    StringRef Directory;
  };

  /// Map of local ID to SourceLocation.
  DenseMap<uint64_t, SourceLocation> LocalIdToSourceLocationMap;

  /// Create a struct type to match the "struct SourceLocation" type.
  /// (and the source_loc_t type in csi.h).
  static StructType *getSourceLocStructType(LLVMContext &C);

  /// Append the line and file information to the table.
  void add(uint64_t ID, int32_t Line = -1,
           StringRef Filename = "", StringRef Directory = "",
           StringRef Name = "");
};

namespace {
struct CilkSanitizerImpl : public CSIImpl {
  // Class to manage inserting instrumentation without static race detection.
  class SimpleInstrumentor {
  public:
    SimpleInstrumentor(CilkSanitizerImpl &CilkSanImpl, TaskInfo &TI,
                       LoopInfo &LI, DominatorTree &DT,
                       const TargetLibraryInfo *TLI)
        : CilkSanImpl(CilkSanImpl), TI(TI), LI(LI), DT(DT),
          DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy), TLI(TLI) {}

    bool instrumentSimpleInstructions(
        SmallVectorImpl<Instruction *> &Instructions);
    bool instrumentAnyMemIntrinsics(
        SmallVectorImpl<Instruction *> &MemIntrinsics);
    bool instrumentCalls(SmallVectorImpl<Instruction *> &Calls);
    bool instrumentAncillaryInstructions(
        SmallPtrSetImpl<Instruction *> &Allocas,
        SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
        SmallPtrSetImpl<Instruction *> &FreeCalls,
        DenseMap<Value *, unsigned> &SyncRegNums,
        DenseMap<BasicBlock *, unsigned> &SRCounters, const DataLayout &DL);

  private:
    void getDetachesForInstruction(Instruction *I);

    CilkSanitizerImpl &CilkSanImpl;
    TaskInfo &TI;
    LoopInfo &LI;
    DominatorTree &DT;
    DomTreeUpdater DTU;
    const TargetLibraryInfo *TLI;

    SmallSetVector<DetachInst *, 8> Detaches;
  };

  // Class to manage inserting instrumentation with static race detection.
  class Instrumentor {
  public:
    Instrumentor(CilkSanitizerImpl &CilkSanImpl, RaceInfo &RI, TaskInfo &TI,
                 LoopInfo &LI, DominatorTree &DT, const TargetLibraryInfo *TLI)
        : CilkSanImpl(CilkSanImpl), RI(RI), TI(TI), LI(LI), DT(DT),
          DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy), TLI(TLI) {}

    void insertArgMAAPs(Function &F, Value *FuncId);
    bool instrumentSimpleInstructions(
        SmallVectorImpl<Instruction *> &Instructions);
    bool instrumentAnyMemIntrinsics(
        SmallVectorImpl<Instruction *> &MemIntrinsics);
    bool instrumentCalls(SmallVectorImpl<Instruction *> &Calls);
    void getDetachesForCoalescedInstrumentation(
        SmallPtrSetImpl<Instruction *> &LoopInstToHoist,
        SmallPtrSetImpl<Instruction *> &LoopInstToSink);
    bool instrumentAncillaryInstructions(
        SmallPtrSetImpl<Instruction *> &Allocas,
        SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
        SmallPtrSetImpl<Instruction *> &FreeCalls,
        DenseMap<Value *, unsigned> &SyncRegNums,
        DenseMap<BasicBlock *, unsigned> &SRCounters, const DataLayout &DL);
    bool instrumentLoops(SmallPtrSetImpl<Instruction *> &LoopInstToHoist,
                         SmallPtrSetImpl<Instruction *> &LoopInstToSink,
                         SmallPtrSetImpl<const Loop *> &TapirLoops,
                         ScalarEvolution *);
    bool performDelayedInstrumentation();

  private:
    void getDetachesForInstruction(Instruction *I);
    // A MAAP (May Access Alias in Parallel) encodes static information about
    // memory access that may result in a race, in order to propagate that
    // information dynamically at runtime.  In particular, a MAAP for a pointer
    // argument to a called function communicates to the callee whether the
    // caller or some ancestor may read or write the referenced memory in
    // parallel and whether the caller can provide any noalias guarantee on that
    // memory location.
    enum class MAAPValue : uint8_t
      {
       NoAccess = 0,
       Mod = 1,
       Ref = 2,
       ModRef = Mod | Ref,
       NoAlias = 4,
      };
    static unsigned raceTypeToFlagVal(RaceInfo::RaceType RT);
    // Get the MAAP value for specific instruction and operand.
    Value *getMAAPValue(Instruction *I, IRBuilder<> &IRB,
                        unsigned OperandNum = static_cast<unsigned>(-1),
                        MAAPValue DefaultMV = MAAPValue::ModRef,
                        bool CheckArgs = true);
    // Helper method to determine noalias MAAP bit.
    Value *getNoAliasMAAPValue(Instruction *I, IRBuilder<> &IRB,
                               unsigned OperandNum, MemoryLocation Loc,
                               const RaceInfo::RaceData &RD,
                               const Value *Obj, Value *MAAPVal);
    // Synthesize a check of the MAAP to determine whether the MAAP means we can
    // skip executing instrumentation for the given instruction.
    Value *getMAAPCheck(Instruction *I, IRBuilder<> &IRB,
                        unsigned OperandNum = static_cast<unsigned>(-1));
    // Helper method to read a MAAP value.
    Value *readMAAPVal(Value *V, IRBuilder<> &IRB);

    CilkSanitizerImpl &CilkSanImpl;
    RaceInfo &RI;
    TaskInfo &TI;
    LoopInfo &LI;
    DominatorTree &DT;
    DomTreeUpdater DTU;
    const TargetLibraryInfo *TLI;

    SmallSetVector<DetachInst *, 8> Detaches;

    DenseMap<const Value *, Value *> LocalMAAPs;
    SmallPtrSet<const Value *, 8> ArgMAAPs;

    SmallVector<Instruction *, 8> DelayedSimpleInsts;
    SmallVector<std::pair<Instruction *, unsigned>, 8> DelayedMemIntrinsics;
    SmallVector<Instruction *, 8> DelayedCalls;
  };

  // TODO: With recent changes in LLVM's JIT technology, the JitMode option
  // no longer seems to be necessary.
  CilkSanitizerImpl(Module &M, CallGraph *CG,
                    function_ref<DominatorTree &(Function &)> GetDomTree,
                    function_ref<TaskInfo &(Function &)> GetTaskInfo,
                    function_ref<LoopInfo &(Function &)> GetLoopInfo,
                    function_ref<RaceInfo &(Function &)> GetRaceInfo,
                    function_ref<TargetLibraryInfo &(Function &)> GetTLI,
                    function_ref<ScalarEvolution &(Function &)> GetSE,
                    // function_ref<TargetTransformInfo &(Function &)> GetTTI,
                    bool CallsMayThrow = !AssumeNoExceptions,
                    bool JitMode = false)
      : CSIImpl(M, CG, GetDomTree, GetLoopInfo, GetTaskInfo, GetTLI, GetSE,
                nullptr),
        GetRaceInfo(GetRaceInfo) {
    // Even though we're doing our own instrumentation, we want the CSI setup
    // for the instrumentation of function entry/exit, memory accesses (i.e.,
    // loads and stores), atomics, memory intrinsics.  We also want call sites,
    // for extracting debug information.
    Options.InstrumentBasicBlocks = false;
    Options.InstrumentLoops = true;
    // Cilksan defines its own hooks for instrumenting memory accesses, memory
    // intrinsics, and Tapir instructions, so we disable the default CSI
    // instrumentation hooks for these IR objects.
    Options.InstrumentMemoryAccesses = false;
    Options.InstrumentMemIntrinsics = false;
    Options.InstrumentTapir = false;
    Options.InstrumentCalls = false;
    Options.jitMode = JitMode;
    Options.CallsMayThrow = CallsMayThrow;
  }
  bool setup(bool NeedToSetupCalls);
  bool run();

  static StructType *getUnitObjTableType(LLVMContext &C,
                                         PointerType *EntryPointerType);
  static Constant *objTableToUnitObjTable(Module &M,
                                          StructType *UnitObjTableType,
                                          ObjectTable &ObjTable);
  static bool isLibCall(const Instruction &I, const TargetLibraryInfo *TLI);
  static bool simpleCallCannotRace(const Instruction &I);
  static bool shouldIgnoreCall(const Instruction &I);

  static DebugLoc searchForDebugLoc(Instruction *I) {
    if (DebugLoc Loc = I->getDebugLoc())
      return Loc;

    // Try to find debug information later in this block.
    BasicBlock::iterator BI = I->getIterator();
    BasicBlock::const_iterator BE(I->getParent()->end());
    while (BI != BE) {
      if (DebugLoc Loc = BI->getDebugLoc()) {
        return Loc;
      }
      ++BI;
    }

    // Try to find debug information earlier in this block.
    BI = I->getIterator();
    BasicBlock::const_iterator BB(I->getParent()->begin());
    while (BI != BB) {
      --BI;
      if (DebugLoc Loc = BI->getDebugLoc()) {
        return Loc;
      }
    }

    return I->getDebugLoc();
  }

  void setupBlocks(Function &F, DominatorTree *DT = nullptr,
                   LoopInfo *LI = nullptr);
  bool setupFunction(Function &F, bool NeedToSetupCalls);

  // Methods for handling FED tables
  void initializeFEDTables() {}
  void collectUnitFEDTables() {}

  // Methods for handling object tables
  void initializeCsanObjectTables();
  void collectUnitObjectTables();

  // Create a call to the runtime unit initialization routine in a global
  // constructor.
  CallInst *createRTUnitInitCall(IRBuilder<> &IRB) override;

  // Initialize custom hooks for CilkSanitizer
  void initializeCsanHooks();

  Value *getCalleeFuncID(const Function *Callee, IRBuilder<> &IRB);

  // Helper function for prepareToInstrumentFunction that chooses loads and
  // stores in a basic block to instrument.
  void chooseInstructionsToInstrument(SmallVectorImpl<Instruction *> &Local,
                                      SmallVectorImpl<Instruction *> &All,
                                      const TaskInfo &TI, LoopInfo &LI,
                                      const TargetLibraryInfo *TLI);

  // Helper methods for instrumenting different IR objects.
  bool instrumentLoadOrStore(Instruction *I, IRBuilder<> &IRB);
  bool instrumentLoadOrStore(Instruction *I) {
    IRBuilder<> IRB(I);
    if (!IRB.getCurrentDebugLocation())
      IRB.SetCurrentDebugLocation(searchForDebugLoc(I));
    return instrumentLoadOrStore(I, IRB);
  }
  bool instrumentAtomic(Instruction *I, IRBuilder<> &IRB);
  bool instrumentAtomic(Instruction *I) {
    IRBuilder<> IRB(I);
    if (!IRB.getCurrentDebugLocation())
      IRB.SetCurrentDebugLocation(searchForDebugLoc(I));
    return instrumentAtomic(I, IRB);
  }
  bool instrumentIntrinsicCall(Instruction *I,
                               SmallVectorImpl<Value *> *MAAPVals = nullptr);
  bool instrumentLibCall(Instruction *I,
                         SmallVectorImpl<Value *> *MAAPVals = nullptr);
  bool instrumentCallsite(Instruction *I,
                          SmallVectorImpl<Value *> *MAAPVals = nullptr);
  bool suppressCallsite(Instruction *I);
  bool instrumentAllocFnLibCall(Instruction *I, const TargetLibraryInfo *TLI);
  bool instrumentAllocationFn(Instruction *I, DominatorTree &DT,
                              const TargetLibraryInfo *TLI);
  bool instrumentFree(Instruction *I, const TargetLibraryInfo *TLI);
  bool instrumentDetach(DetachInst *DI, unsigned SyncRegNum,
                        unsigned NumSyncRegs, DominatorTree &DT, TaskInfo &TI,
                        LoopInfo &LI);
  bool instrumentSync(SyncInst *SI, unsigned SyncRegNum);
  void instrumentTapirLoop(Loop &L, TaskInfo &TI,
                      DenseMap<Value *, unsigned> &SyncRegNums,
                      ScalarEvolution *SE = nullptr);
  bool instrumentAlloca(Instruction *I, TaskInfo &TI);

  bool instrumentFunctionUsingRI(Function &F);
  // Helper method for RI-based race detection for instrumenting an access by a
  // memory intrinsic.
  bool instrumentAnyMemIntrinAcc(Instruction *I, unsigned OperandNum,
                                 IRBuilder<> &IRB);
  bool instrumentAnyMemIntrinAcc(Instruction *I, unsigned OperandNum) {
    IRBuilder<> IRB(I);
    if (!IRB.getCurrentDebugLocation())
      IRB.SetCurrentDebugLocation(searchForDebugLoc(I));
    return instrumentAnyMemIntrinAcc(I, OperandNum, IRB);
  }

  // A range check of Len bytes from Addr, as a read or write like Rep: hoisted
  // before loop nest O (see HoistNests), or, for straight-line accesses (see
  // CoalesceBlocks), inserted before At, after the last access it covers.
  struct NestCheck {
    Instruction *Rep;
    Loop *O;
    const SCEV *Addr;
    uint64_t Len;
    Instruction *At = nullptr;
    // Symbolic nests (see HoistSymbolicNests): the length is LenS, and the
    // check repeats over each (stride, count) in Rows, outermost first.
    const SCEV *LenS = nullptr;
    SmallVector<std::pair<const SCEV *, const SCEV *>, 2> Rows{};
    // Conditions fixed on entry to O, and the values under which the
    // accesses run; the check runs only when all of them hold.
    SmallVector<std::pair<Value *, bool>, 2> Preds{};
    // The accesses whose bytes the check covers. With MAAP checks on, it is
    // skipped only when each of them could skip its own check: their racers
    // can differ from Rep's.
    SmallVector<Instruction *, 4> Members{};
    // A footprint marker's check (see FootprintMarker): its address, length
    // and rows are already computed before At, and O is null.
    Value *AddrV = nullptr, *LenV = nullptr;
    SmallVector<std::pair<Value *, Value *>, 2> RowsV{};
    Instruction *insertPt() const {
      return At ? At : O->getLoopPreheader()->getTerminator();
    }
  };
  SmallVector<NestCheck, 8> NestChecks;
  void planBlockChecks(Function &F, ScalarEvolution &SE, RaceInfo &RI,
                       SmallVectorImpl<Instruction *> &LoadsAndStores);
  void planNestChecks(LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT,
                      TaskInfo &TI, RaceInfo &RI,
                      SmallPtrSetImpl<Instruction *> &Covered);
  static const char *whyNotOneStrand(const Loop *O, TaskInfo &TI);
  // An access of a class (see coverClass): Size bytes at Off from the class's
  // base, repeated over each (step, count) in Steps, innermost loop first;
  // none for a straight-line access.
  struct FootprintAccess {
    Instruction *I;
    int64_t Off;
    int64_t Size;
    SmallVector<std::pair<int64_t, unsigned>, 4> Steps{};
  };
  unsigned coverClass(ArrayRef<FootprintAccess> As, ScalarEvolution &SE,
                      const SCEV *Base, Loop *O, Instruction *At,
                      bool FirstRep, int64_t MaxSpan, uint64_t MaxMarks,
                      unsigned MaxRuns);
  bool tryNest(Loop *O, LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT,
               TaskInfo &TI, RaceInfo &RI,
               SmallPtrSetImpl<Instruction *> &Covered);
  // A symbolic nest's footprint (see symbolicFootprints): Base + [0, Len) +
  // the sum of Step_k * [0, Count_k) over Rows, outermost first, accessed
  // when Preds hold, by Members. Dup marks a footprint another one's check
  // covers: the same, and a store's or earlier. A footprint that isn't Dup
  // also has the members of its duplicates.
  struct SymFootprint {
    Instruction *I;
    const SCEV *Base, *Len;
    SmallVector<std::pair<const SCEV *, const SCEV *>, 2> Rows;
    SmallVector<std::pair<Value *, bool>, 2> Preds;
    SmallVector<Instruction *, 4> Members;
    bool Partly;
    bool Dup = false;
  };
  static bool
  symbolicFootprints(Loop *O, LoopInfo &LI, ScalarEvolution &SE,
                     DominatorTree &DT, TaskInfo &TI,
                     function_ref<std::optional<int64_t>(Instruction &)> SizeOf,
                     SmallVectorImpl<SymFootprint> &Prints,
                     DenseMap<Instruction *, unsigned> &NumVersions);
  static unsigned placeFootprintMarkers(Function &F, LoopInfo &LI,
                                        ScalarEvolution &SE, DominatorTree &DT,
                                        TaskInfo &TI);
  // A footprint marker of a function to instrument, taken out of it before
  // any analysis sees it: the values it was called with, where its checks go
  // (At, where it was), and the accesses it covers, each in one strand with
  // At and after it (see takeFootprintMarkers).
  struct FootprintMarker {
    Instruction *At;
    WeakTrackingVH Guard, Base, Len;
    SmallVector<std::pair<WeakTrackingVH, WeakTrackingVH>, 2> Rows;
    SmallVector<Instruction *, 8> Members;
  };
  MapVector<Function *, SmallVector<FootprintMarker, 4>> FootprintMarkers;
  // The values markers were called with, deleted at the end if unused.
  SmallVector<WeakTrackingVH, 16> FootprintOperands;
  void takeFootprintMarkers(Function &F, DominatorTree &DT, TaskInfo &TI);
  void planFootprintChecks(Function &F, ScalarEvolution &SE, RaceInfo &RI,
                           SmallPtrSetImpl<Instruction *> &Covered);
  void removeFootprintMarkers();

  bool instrumentLoadOrStoreHoisted(Instruction *I,
                                    Value *Addr,
                                    Value *RangeVal,
                                    IRBuilder<> &IRB,
                                    uint64_t LocalId,
                                    MaybeAlign Alignment = std::nullopt);

private:
  // Analysis results
  function_ref<RaceInfo &(Function &)> GetRaceInfo;

  // Instrumentation hooks
  FunctionCallee CsanFuncEntry = nullptr;
  FunctionCallee CsanFuncExit = nullptr;
  FunctionCallee CsanRead = nullptr;
  FunctionCallee CsanWrite = nullptr;
  FunctionCallee CsanLargeRead = nullptr;
  FunctionCallee CsanLargeWrite = nullptr;
  FunctionCallee GetCurOSLabel = nullptr;

  // Per-function state for hoisting the current-OS-label fetch. The label is
  // invariant within a strand, and a Tapir spindle is exactly a maximal strand
  // (spindle boundaries are detach/reattach/sync), so every instrumentation
  // site in a spindle can share one fetch instead of making its own. Without
  // this, instrumenting a loop body re-fetches the label every iteration:
  // cholesky emitted 436 fetches, and the call is not cheap.
  TaskInfo *CurTaskInfo = nullptr;
  DominatorTree *CurDomTree = nullptr;
  LoopInfo *CurLoopInfo = nullptr;
  DenseMap<const BasicBlock *, CallInst *> SpindleLabelCache;

  // Returns the label fetch to use at IRB's insertion point, creating one at
  // the enclosing spindle's entry the first time that spindle needs it.
  CallInst *getCurOSLabelCall(IRBuilder<> &IRB);
public:
  // While set, getCurOSLabelCall returns this fetch: the loops of row checks
  // before a symbolic nest are new blocks that TaskInfo doesn't know, in the
  // strand of the block they were split from.
  CallInst *LabelOverride = nullptr;
private:

  bool LoadTakesLabel = false;
  bool StoreTakesLabel = false;
  bool LargeLoadTakesLabel = false;
  bool LargeStoreTakesLabel = false;
  FunctionCallee CsanBeforeCallsite = nullptr;
  FunctionCallee CsanAfterCallsite = nullptr;
  FunctionCallee CsanDetach = nullptr;
  FunctionCallee CsanDetachContinue = nullptr;
  FunctionCallee CsanTaskEntry = nullptr;
  FunctionCallee CsanTaskExit = nullptr;
  FunctionCallee CsanAfterSync = nullptr;
  FunctionCallee CsanBeforeSync = nullptr;
  FunctionCallee CsanBeforeLoop = nullptr;
  FunctionCallee CsanAfterLoop = nullptr;
  FunctionCallee CsanBeforeAllocFn = nullptr;
  FunctionCallee CsanAfterAllocFn = nullptr;
  FunctionCallee CsanBeforeFree = nullptr;
  FunctionCallee CsanAfterFree = nullptr;

  // Hooks for suppressing instrumentation, e.g., around callsites that cannot
  // expose a race.
  FunctionCallee CsanDisableChecking = nullptr;
  FunctionCallee CsanEnableChecking = nullptr;

  FunctionCallee GetMAAP = nullptr;
  FunctionCallee SetMAAP = nullptr;

  // CilkSanitizer custom forensic tables
  ObjectTable LoadObj, StoreObj, AllocaObj, AllocFnObj;

  SmallVector<Constant *, 4> UnitObjTables;

  SmallVector<Instruction *, 8> AllocationFnCalls;
  SmallVector<Instruction *, 8> FreeCalls;
  SmallVector<Instruction *, 8> Allocas;
  SmallPtrSet<Instruction *, 8> ToInstrument;

  // Map of functions to updated race type, for interprocedural analysis of
  // races.
  DenseMap<const Function *, RaceInfo::RaceType> FunctionRaceType;
  DenseMap<const Value *, ModRefInfo> ObjectMRForRace;

  DenseMap<DetachInst *, SmallVector<SyncInst *, 2>> DetachToSync;

  bool localBaseObj(const Value *Addr, LoopInfo *LI,
                    const TargetLibraryInfo *TLI) const;
  bool possibleRaceByCapture(const Value *Addr, const TaskInfo &TI,
                             LoopInfo *LI) const;
  bool unknownObjectUses(const Value *Addr, LoopInfo *LI,
                         const TargetLibraryInfo *TLI) const;

  // Cached results of calls to getUnderlyingObjects.
  using BaseObjMapTy =
      DenseMap<const Value *, SmallVector<const Value *, 1>>;
  mutable BaseObjMapTy BaseObjects;
  SmallVectorImpl<const Value *> &lookupBaseObjects(const Value *Addr,
                                                    LoopInfo *LI) const {
    if (!BaseObjects.count(Addr)) {
      if (isa<GlobalValue>(Addr))
        BaseObjects.lookup(Addr);
      else
        getUnderlyingObjects(Addr, BaseObjects[Addr], LI, 0);
    }
    return BaseObjects[Addr];
  }

  bool mightHaveDetachedUse(const Value *Addr, const TaskInfo &TI) const;
  // // Cached results of calls to MightHaveDetachedUse.
  // using DetachedUseMapTy = DenseMap<const Value *, bool>;
  // mutable DetachedUseMapTy DetachedUseCache;
  bool lookupMightHaveDetachedUse(const Value *Addr, const TaskInfo &TI) const {
    return mightHaveDetachedUse(Addr, TI);
    // if (!DetachedUseCache.count(Addr))
    //   DetachedUseCache[Addr] = MightHaveDetachedUse(Addr, TI);
    // return DetachedUseCache[Addr];
  }

  // Cached results of calls to PointerMayBeCaptured.
  using MayBeCapturedMapTy = DenseMap<const Value *, bool>;
  mutable MayBeCapturedMapTy MayBeCapturedCache;
  bool lookupPointerMayBeCaptured(const Value *Ptr) const {
    if (!Ptr->getType()->isPointerTy())
      return false;

    if (!MayBeCapturedCache.count(Ptr)) {
      if (isa<GlobalValue>(Ptr))
        MayBeCapturedCache.lookup(Ptr);
      else
        MayBeCapturedCache[Ptr] =
            PointerMayBeCaptured(Ptr, true, MaxUsesToExploreCapture);
    }
    return MayBeCapturedCache[Ptr];
  }

  FunctionCallee getOrInsertSynthesizedHook(StringRef Name, FunctionType *T,
                                            int ReturnParam = -1,
                                            AttributeList AL = AttributeList());
};

/// CilkSanitizer: instrument the code in module to find races.
struct CilkSanitizerLegacyPass : public ModulePass {
  static char ID;  // Pass identification, replacement for typeid.
  CilkSanitizerLegacyPass(bool CallsMayThrow = !AssumeNoExceptions,
                          bool JitMode = false)
      : ModulePass(ID), JitMode(JitMode), CallsMayThrow(CallsMayThrow) {
    initializeCilkSanitizerLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override { return "CilkSanitizer"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;

  bool JitMode = false;
  bool CallsMayThrow = true;
};
} // end anonymous namespace

char CilkSanitizerLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    CilkSanitizerLegacyPass, "csan",
    "CilkSanitizer: detects determinacy races in Cilk programs.",
    false, false)
INITIALIZE_PASS_DEPENDENCY(BasicAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TapirRaceDetectWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(
    CilkSanitizerLegacyPass, "csan",
    "CilkSanitizer: detects determinacy races in Cilk programs.",
    false, false)

void CilkSanitizerLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<TapirRaceDetectWrapperPass>();
  AU.addRequired<TaskInfoWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
}

uint64_t ObjectTable::add(Instruction &I, Value *Obj) {
  uint64_t ID = getId(&I);
  // An access a symbolic nest covers only partly is registered twice: for
  // its own check and as the representative of the nest's checks.
  if (LocalIdToSourceLocationMap.count(ID))
    return ID;
  if (isa<UndefValue>(Obj)) {
    add(ID, -1, "", "", "(undef)");
    return ID;
  }

  // First, if the underlying object is a global variable, get that variable's
  // debug information.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Obj)) {
    SmallVector<DIGlobalVariableExpression *, 1> DbgGVExprs;
    GV->getDebugInfo(DbgGVExprs);
    for (auto *GVE : DbgGVExprs) {
      auto *DGV = GVE->getVariable();
      if (DGV->getName() != "") {
        add(ID, DGV->getLine(), DGV->getFilename(), DGV->getDirectory(),
            DGV->getName());
        return ID;
      }
    }
    add(ID, -1, "", "", Obj->getName());
    return ID;
  }

  // Otherwise, if the underlying object is a function, get that function's
  // debug information.
  if (Function *F = dyn_cast<Function>(Obj)) {
    if (DISubprogram *SP = F->getSubprogram()) {
      add(ID, SP->getLine(), SP->getFilename(), SP->getDirectory(),
          SP->getName());
      return ID;
    }
    add(ID, -1, "", "", Obj->getName());
    return ID;
  }

  // Next, if this is an alloca instruction, look for a #dbg_declare.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Obj)) {
    TinyPtrVector<DbgDeclareInst *> DbgDeclares = findDbgDeclares(AI);
    if (!DbgDeclares.empty()) {
      auto *LV = DbgDeclares.front()->getVariable();
      add(ID, LV->getLine(), LV->getFilename(), LV->getDirectory(),
          LV->getName());
      return ID;
    }
  }

  // Otherwise look for a #dbg_value.
  SmallVector<DbgVariableRecord *> DbgVariableRecords;
  SmallVector<DbgValueInst *, 1> DbgValues;
  findDbgValues(DbgValues, Obj, &DbgVariableRecords);
  for (auto *DVI : DbgValues) {
    auto *LV = DVI->getVariable();
    if (LV->getName() != "") {
      add(ID, LV->getLine(), LV->getFilename(), LV->getDirectory(),
          LV->getName());
      return ID;
    }
  }
  for (auto *DVR : DbgVariableRecords) {
    auto *LV = DVR->getVariable();
    if (LV->getName() != "") {
      add(ID, LV->getLine(), LV->getFilename(), LV->getDirectory(),
          LV->getName());
      return ID;
    }
  }

  add(ID, -1, "", "", Obj->getName());
  return ID;
}

PointerType *ObjectTable::getPointerType(LLVMContext &C) {
  return PointerType::get(C, 0);
}

StructType *ObjectTable::getSourceLocStructType(LLVMContext &C) {
  return StructType::get(
      /* Name */ PointerType::get(C, 0),
      /* Line */ IntegerType::get(C, 32),
      /* File */ PointerType::get(C, 0));
}

void ObjectTable::add(uint64_t ID, int32_t Line,
                      StringRef Filename, StringRef Directory,
                      StringRef Name) {
  assert(LocalIdToSourceLocationMap.find(ID) ==
             LocalIdToSourceLocationMap.end() &&
         "Id already exists in FED table.");
  LocalIdToSourceLocationMap[ID] = {Name, Line, Filename, Directory};
}

// The order of arguments to ConstantStruct::get() must match the
// obj_source_loc_t type in csan.h.
static void addObjTableEntries(SmallVectorImpl<Constant *> &TableEntries,
                               StructType *TableType, Constant *Name,
                               Constant *Line, Constant *File) {
  TableEntries.push_back(ConstantStruct::get(TableType, Name, Line, File));
}

Constant *ObjectTable::insertIntoModule(Module &M) const {
  LLVMContext &C = M.getContext();
  StructType *TableType = getSourceLocStructType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  SmallVector<Constant *, 6> TableEntries;

  // Get the object-table entries for each ID.
  for (uint64_t LocalID = 0; LocalID < IdCounter; ++LocalID) {
    const SourceLocation &E = LocalIdToSourceLocationMap.find(LocalID)->second;
    // Source line
    Constant *Line = ConstantInt::get(Int32Ty, E.Line);
    // Source file
    Constant *File;
    {
      std::string Filename = E.Filename.str();
      if (!E.Directory.empty())
        Filename = E.Directory.str() + "/" + Filename;
      File = getObjectStrGV(M, Filename, "__csi_unit_filename_");
    }
    // Variable name
    Constant *Name = getObjectStrGV(M, E.Name, "__csi_unit_object_name_");

    // Add entry to the table
    addObjTableEntries(TableEntries, TableType, Name, Line, File);
  }

  ArrayType *TableArrayType = ArrayType::get(TableType, TableEntries.size());
  Constant *Table = ConstantArray::get(TableArrayType, TableEntries);
  GlobalVariable *GV =
    new GlobalVariable(M, TableArrayType, false, GlobalValue::InternalLinkage,
                       Table, CsiUnitObjTableName);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

namespace {

using SCCNodeSet = SmallSetVector<Function *, 8>;

} // end anonymous namespace

bool CilkSanitizerImpl::setup(bool NeedToSetupCalls) {
  // Setup functions for instrumentation.
  for (scc_iterator<CallGraph *> I = scc_begin(CG); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &SCC = *I;
    for (CallGraphNode *N : SCC)
      if (Function *F = N->getFunction())
        setupFunction(*F, NeedToSetupCalls);
  }
  return true;
}

bool CilkSanitizerImpl::run() {
  // Link the tool bitcode once initially, to get type definitions.
  linkInToolFromBitcode(ClCilksanBCPath);
  // Initialize components of the CSI and Cilksan system.
  initializeCsi();
  initializeFEDTables();
  initializeCsanObjectTables();
  initializeCsanHooks();

  // Take out the footprint markers, placing them first where the early pass
  // didn't run (see CilksanFootprintsPass), before any analysis sees them.
  if (EnableStaticRaceDetection && LoopHoisting && HoistNests &&
      HoistSymbolicNests)
    for (Function &F : M)
      if (!F.empty() && !shouldNotInstrumentFunction(F) &&
          !LinkedFromBitcode.count(&F) && checkSanitizeCilkAttr(F) &&
          !F.hasFnAttribute(FootprintsAttr))
        NumFootprintMarkers += placeFootprintMarkers(
            F, GetLoopInfo(F), (*GetScalarEvolution)(F), GetDomTree(F),
            GetTaskInfo(F));
  if (Function *Fn = M.getFunction(FootprintMarkerName)) {
    SmallPtrSet<Function *, 8> HasMarkers;
    for (User *U : Fn->users())
      if (auto *Call = dyn_cast<CallInst>(U))
        HasMarkers.insert(Call->getFunction());
    for (Function &F : M)
      if (HasMarkers.count(&F))
        takeFootprintMarkers(F, GetDomTree(F), GetTaskInfo(F));
  }

  // Evaluate the SCC's in the callgraph in post order to support
  // interprocedural analysis of potential races in the module.
  SmallVector<Function *, 16> InstrumentedFunctions;

  // Instrument functions.
  for (scc_iterator<CallGraph *> I = scc_begin(CG); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &SCC = *I;
    for (CallGraphNode *N : SCC) {
      if (Function *F = N->getFunction())
        if (instrumentFunctionUsingRI(*F))
          InstrumentedFunctions.push_back(F);
    }
  }
  // After all functions have been analyzed and instrumented, update their
  // attributes.
  for (Function *F : InstrumentedFunctions) {
    updateInstrumentedFnAttrs(*F);
    F->removeFnAttr(Attribute::SanitizeCilk);
  }
  removeFootprintMarkers();

  CSIImpl::collectUnitFEDTables();
  collectUnitFEDTables();
  collectUnitObjectTables();
  finalizeCsi();

  // Link the tool bitcode a second time, for definitions of used functions.
  linkInToolFromBitcode(ClCilksanBCPath);
  return true;
}

void CilkSanitizerImpl::initializeCsanObjectTables() {
  LoadObj = ObjectTable(M, CsiLoadBaseIdName);
  StoreObj = ObjectTable(M, CsiStoreBaseIdName);
  AllocaObj = ObjectTable(M, CsiAllocaBaseIdName);
  AllocFnObj = ObjectTable(M, CsiAllocFnBaseIdName);
}

// Create a struct type to match the unit_obj_entry_t type in csanrt.c.
StructType *CilkSanitizerImpl::getUnitObjTableType(
    LLVMContext &C, PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64), EntryPointerType);
}

Constant *CilkSanitizerImpl::objTableToUnitObjTable(
    Module &M, StructType *UnitObjTableType, ObjectTable &ObjTable) {
  Constant *NumEntries =
    ConstantInt::get(IntegerType::get(M.getContext(), 64), ObjTable.size());
  // Constant *BaseIdPtr =
  //   ConstantExpr::getPointerCast(FedTable.baseId(),
  //                                Type::getInt8PtrTy(M.getContext(), 0));
  Constant *InsertedTable = ObjTable.insertIntoModule(M);
  return ConstantStruct::get(UnitObjTableType, NumEntries,
                             InsertedTable);
}

void CilkSanitizerImpl::collectUnitObjectTables() {
  LLVMContext &C = M.getContext();
  StructType *UnitObjTableType =
      getUnitObjTableType(C, ObjectTable::getPointerType(C));

  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, LoadObj));
  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, StoreObj));
  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, AllocaObj));
  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, AllocFnObj));
}

CallInst *CilkSanitizerImpl::createRTUnitInitCall(IRBuilder<> &IRB) {
  LLVMContext &C = M.getContext();

  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));
  StructType *UnitObjTableType =
      getUnitObjTableType(C, ObjectTable::getPointerType(C));

  // Lookup __csanrt_unit_init
  SmallVector<Type *, 4> InitArgTypes(
      {PointerType::getUnqual(C), PointerType::get(C, 0),
       PointerType::get(C, 0), InitCallsiteToFunction->getType()});
  FunctionType *InitFunctionTy =
      FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  RTUnitInit = M.getOrInsertFunction(CsanRtUnitInitName, InitFunctionTy);
  assert(isa<Function>(RTUnitInit.getCallee()) &&
         "Failed to get or insert __csanrt_unit_init function");

  ArrayType *UnitFedTableArrayType =
      ArrayType::get(UnitFedTableType, UnitFedTables.size());
  Constant *FEDTable = ConstantArray::get(UnitFedTableArrayType, UnitFedTables);
  GlobalVariable *FEDGV = new GlobalVariable(M, UnitFedTableArrayType, false,
                                             GlobalValue::InternalLinkage, FEDTable,
                                             CsiUnitFedTableArrayName);

  ArrayType *UnitObjTableArrayType =
      ArrayType::get(UnitObjTableType, UnitObjTables.size());
  Constant *ObjTable = ConstantArray::get(UnitObjTableArrayType, UnitObjTables);
  GlobalVariable *ObjGV = new GlobalVariable(M, UnitObjTableArrayType, false,
                                             GlobalValue::InternalLinkage, ObjTable,
                                             CsiUnitObjTableArrayName);

  Constant *Zero = ConstantInt::get(IRB.getInt32Ty(), 0);
  Value *GepArgs[] = {Zero, Zero};

  // Insert call to __csanrt_unit_init
  return IRB.CreateCall(
      RTUnitInit,
      {IRB.CreateGlobalString(M.getName()),
       ConstantExpr::getGetElementPtr(FEDGV->getValueType(), FEDGV, GepArgs),
       ConstantExpr::getGetElementPtr(ObjGV->getValueType(), ObjGV, GepArgs),
       InitCallsiteToFunction});
}

// Initialize all instrumentation hooks that are specific to CilkSanitizer.
void CilkSanitizerImpl::initializeCsanHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *FuncPropertyTy = CsiFuncProperty::getType(C);
  Type *FuncExitPropertyTy = CsiFuncExitProperty::getType(C);
  Type *TaskPropertyTy = CsiTaskProperty::getType(C);
  Type *TaskExitPropertyTy = CsiTaskExitProperty::getType(C);
  Type *LoadPropertyTy = CsiLoadStoreProperty::getType(C);
  Type *StorePropertyTy = CsiLoadStoreProperty::getType(C);
  Type *CallPropertyTy = CsiCallProperty::getType(C);
  Type *LoopPropertyTy = CsiLoopProperty::getType(C);
  Type *AllocFnPropertyTy = CsiAllocFnProperty::getType(C);
  Type *FreePropertyTy = CsiFreeProperty::getType(C);
  Type *DetachPropertyTy = CsiDetachProperty::getType(C);
  Type *DetContPropertyTy = CsiDetachContinueProperty::getType(C);
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getPtrTy();
  Type *NumBytesType = IRB.getInt32Ty();
  Type *LargeNumBytesType = IntptrTy;
  Type *IDType = IRB.getInt64Ty();
  Type *SyncRegType = IRB.getInt32Ty();

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    FnAttrs = FnAttrs.addParamAttribute(
        C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 2, Attribute::ReadNone);
    CsanFuncEntry = getHookFunction("__csan_func_entry", FnAttrs, RetType,
                                    /* func_id */ IDType,
                                    /* frame_ptr */ AddrType,
                                    /* stack_ptr */ AddrType, FuncPropertyTy);
  }
  {
    CsanFuncExit = getHookFunction("__csan_func_exit", RetType,
                                   /* func_exit_id */ IDType,
                                   /* func_id */ IDType, FuncExitPropertyTy);
  }

  LoadTakesLabel = (FunctionNumParamsInBitcode.lookup("__csan_load") >= 5);
  StoreTakesLabel = (FunctionNumParamsInBitcode.lookup("__csan_store") >= 5);
  LargeLoadTakesLabel = (FunctionNumParamsInBitcode.lookup("__csan_large_load") >= 5);
  LargeStoreTakesLabel = (FunctionNumParamsInBitcode.lookup("__csan_large_store") >= 5);

  if (LoadTakesLabel || StoreTakesLabel || LargeLoadTakesLabel || LargeStoreTakesLabel) {
    FunctionType *GetCurLabTy = FunctionType::get(IRB.getPtrTy(), false);
    AttrBuilder B(C);
    B.addAttribute(Attribute::StrandPure);
    B.addAttribute(Attribute::NoUnwind);
    B.addAttribute(Attribute::WillReturn);
    B.addMemoryAttr(MemoryEffects::inaccessibleMemOnly(ModRefInfo::Ref));
    AttributeList FnAttrs = AttributeList::get(C, AttributeList::FunctionIndex, B);
    GetCurOSLabel = M.getOrInsertFunction("__cilkrts_get_current_os_label", GetCurLabTy, FnAttrs);
    if (Function *Fn = dyn_cast<Function>(GetCurOSLabel.getCallee())) {
      Fn->setCallingConv(CallingConv::PreserveMost);
      Fn->setAttributes(FnAttrs);
    }
  }

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    if (LoadTakesLabel) {
      FnAttrs = FnAttrs.addParamAttribute(
          C, 4, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
      FnAttrs = FnAttrs.addParamAttribute(C, 4, Attribute::ReadOnly);
      CsanRead = getHookFunction("__csan_load", FnAttrs, RetType, IDType,
                                 AddrType, NumBytesType, LoadPropertyTy,
                                 IRB.getPtrTy());
    } else {
      CsanRead = getHookFunction("__csan_load", FnAttrs, RetType, IDType,
                                 AddrType, NumBytesType, LoadPropertyTy);
    }
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    if (StoreTakesLabel) {
      FnAttrs = FnAttrs.addParamAttribute(
          C, 4, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
      FnAttrs = FnAttrs.addParamAttribute(C, 4, Attribute::ReadOnly);
      CsanWrite = getHookFunction("__csan_store", FnAttrs, RetType, IDType,
                                  AddrType, NumBytesType, StorePropertyTy,
                                  IRB.getPtrTy());
    } else {
      CsanWrite = getHookFunction("__csan_store", FnAttrs, RetType, IDType,
                                  AddrType, NumBytesType, StorePropertyTy);
    }
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    if (LargeLoadTakesLabel) {
      FnAttrs = FnAttrs.addParamAttribute(
          C, 4, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
      FnAttrs = FnAttrs.addParamAttribute(C, 4, Attribute::ReadOnly);
      CsanLargeRead =
          getHookFunction("__csan_large_load", FnAttrs, RetType, IDType, AddrType,
                          LargeNumBytesType, LoadPropertyTy, IRB.getPtrTy());
    } else {
      CsanLargeRead =
          getHookFunction("__csan_large_load", FnAttrs, RetType, IDType, AddrType,
                          LargeNumBytesType, LoadPropertyTy);
    }
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    if (LargeStoreTakesLabel) {
      FnAttrs = FnAttrs.addParamAttribute(
          C, 4, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
      FnAttrs = FnAttrs.addParamAttribute(C, 4, Attribute::ReadOnly);
      CsanLargeWrite =
          getHookFunction("__csan_large_store", FnAttrs, RetType, IDType,
                          AddrType, LargeNumBytesType, StorePropertyTy,
                          IRB.getPtrTy());
    } else {
      CsanLargeWrite =
          getHookFunction("__csan_large_store", FnAttrs, RetType, IDType,
                          AddrType, LargeNumBytesType, StorePropertyTy);
    }
  }

  {
    CsanBeforeCallsite = getHookFunction(
        "__csan_before_call", IRB.getVoidTy(), IDType,
        /*callee func_id*/ IDType, IRB.getInt8Ty(), CallPropertyTy);
  }
  {
    CsanAfterCallsite =
        getHookFunction("__csan_after_call", IRB.getVoidTy(), IDType, IDType,
                        IRB.getInt8Ty(), CallPropertyTy);
  }

  {
    CsanDetach = getHookFunction("__csan_detach", RetType,
                                 /* detach_id */ IDType,
                                 /* sync_reg */ SyncRegType, DetachPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 2, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 2, Attribute::ReadNone);
    FnAttrs = FnAttrs.addParamAttribute(
        C, 3, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 3, Attribute::ReadNone);
    CsanTaskEntry = getHookFunction("__csan_task", FnAttrs, RetType,
                                    /* task_id */ IDType,
                                    /* detach_id */ IDType,
                                    /* frame_ptr */ AddrType,
                                    /* stack_ptr */ AddrType, TaskPropertyTy);
  }
  {
    CsanTaskExit =
        getHookFunction("__csan_task_exit", RetType,
                        /* task_exit_id */ IDType,
                        /* task_id */ IDType,
                        /* detach_id */ IDType,
                        /* sync_reg */ SyncRegType, TaskExitPropertyTy);
  }
  {
    CsanDetachContinue =
        getHookFunction("__csan_detach_continue", RetType,
                        /* detach_continue_id */ IDType,
                        /* detach_id */ IDType,
                        /* sync_reg */ SyncRegType, DetContPropertyTy);
  }
  {
    CsanBeforeSync = getHookFunction("__csan_before_sync", RetType, IDType,
                               /* sync_reg */ SyncRegType);
  }
  {
    CsanAfterSync = getHookFunction("__csan_sync", RetType, IDType,
                               /* sync_reg */ SyncRegType);
  }

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    FnAttrs = FnAttrs.addParamAttribute(
        C, 5, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 5, Attribute::ReadNone);
    // Before an allocation function with an old pointer (realloc), so a tool
    // can handle the old block before the allocator can reuse it.
    AttributeList BeforeAttrs;
    BeforeAttrs = BeforeAttrs.addParamAttribute(
        C, 4, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    BeforeAttrs = BeforeAttrs.addParamAttribute(C, 4, Attribute::ReadNone);
    CsanBeforeAllocFn = getHookFunction(
        "__csan_before_allocfn", BeforeAttrs, RetType, IDType,
        /* size */ LargeNumBytesType, /* num elements */ LargeNumBytesType,
        /* alignment */ LargeNumBytesType, /* old ptr */ AddrType,
        /* property */ AllocFnPropertyTy);
    CsanAfterAllocFn = getHookFunction(
        "__csan_after_allocfn", FnAttrs, RetType, IDType,
        /* new ptr */ AddrType, /* size */ LargeNumBytesType,
        /* num elements */ LargeNumBytesType, /* alignment */ LargeNumBytesType,
        /* old ptr */ AddrType, /* property */ AllocFnPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    // Before the free, so a tool can handle the block before the allocator
    // can reuse it.
    CsanBeforeFree =
        getHookFunction("__csan_before_free", FnAttrs, RetType, IDType, AddrType,
                        /* property */ FreePropertyTy);
    CsanAfterFree =
        getHookFunction("__csan_after_free", FnAttrs, RetType, IDType, AddrType,
                        /* property */ FreePropertyTy);
  }

  {
    CsanDisableChecking =
        getHookFunction("__cilksan_disable_checking", RetType);
  }
  {
    CsanEnableChecking = getHookFunction("__cilksan_enable_checking",
                                         RetType);
  }

  Type *MAAPTy = IRB.getInt8Ty();
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addParamAttribute(
        C, 0, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
    GetMAAP = getHookFunction("__csan_get_MAAP", FnAttrs, RetType,
                              PointerType::get(C, 0), IDType, IRB.getInt8Ty());
    // Unlike other hooks, GetMAAP writes to its pointer argument.  Make sure
    // the MemoryEffects on the hook reflect this fact.
    Function *HookFn = cast<Function>(GetMAAP.getCallee());
    HookFn->setMemoryEffects(HookFn->getMemoryEffects() |
                             MemoryEffects::argMemOnly(ModRefInfo::ModRef));
  }
  {
    SetMAAP = getHookFunction("__csan_set_MAAP", RetType, MAAPTy, IDType);
  }

  {
    CsanBeforeLoop = getHookFunction("__csan_before_loop", IRB.getVoidTy(),
                                     IDType, IRB.getInt64Ty(), LoopPropertyTy);
  }
  {
    CsanAfterLoop = getHookFunction("__csan_after_loop", IRB.getVoidTy(),
                                    IDType, IRB.getInt8Ty(), LoopPropertyTy);
  }

  // Cilksan-specific attributes on CSI hooks
  Function *CsiAfterAllocaFn = cast<Function>(CsiAfterAlloca.getCallee());
  CsiAfterAllocaFn->addParamAttr(
      1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
  CsiAfterAllocaFn->addParamAttr(1, Attribute::ReadNone);
  CsiAfterAllocaFn->setOnlyAccessesInaccessibleMemOrArgMem();
  CsiAfterAllocaFn->setDoesNotThrow();
}

static BasicBlock *splitOffPreds(
    BasicBlock *BB, SmallVectorImpl<BasicBlock *> &Preds, DomTreeUpdater *DTU,
    LoopInfo *LI) {
  if (BB->isLandingPad()) {
    SmallVector<BasicBlock *, 2> NewBBs;
    SplitLandingPadPredecessors(BB, Preds, ".csi-split-lp", ".csi-split",
                                NewBBs, DTU, LI);
    return NewBBs[1];
  }

  BasicBlock *NewBB = SplitBlockPredecessors(BB, Preds, ".csi-split", DTU, LI);
  if (isa<UnreachableInst>(BB->getFirstNonPHIOrDbg())) {
    // If the block being split is simply contains an unreachable, then replace
    // the terminator of the new block with an unreachable.  This helps preserve
    // invariants on the CFG structure for Tapir placeholder blocks following
    // detached.rethrow and taskframe.resume terminators.
    ReplaceInstWithInst(NewBB->getTerminator(),
                        new UnreachableInst(BB->getContext()));
    if (DTU) {
      DTU->applyUpdatesPermissive({{DominatorTree::Delete, NewBB, BB}});
    }
  }
  return BB;
}

// Setup each block such that all of its predecessors belong to the same CSI ID
// space.
static void setupBlock(BasicBlock *BB, DomTreeUpdater *DTU, LoopInfo *LI,
                       const TargetLibraryInfo *TLI) {
  if (BB->isLandingPad()) {
    LandingPadInst *LPad = BB->getLandingPadInst();
    if (!LPad->isCleanup())
      LPad->setCleanup(true);
  }

  if (BB->getUniquePredecessor())
    return;

  SmallVector<BasicBlock *, 4> DetachPreds;
  SmallVector<BasicBlock *, 4> TFResumePreds;
  SmallVector<BasicBlock *, 4> SyncPreds;
  SmallVector<BasicBlock *, 4> SyncUnwindPreds;
  SmallVector<BasicBlock *, 4> AllocFnPreds;
  SmallVector<BasicBlock *, 4> FreeFnPreds;
  MapVector<const Function *, SmallVector<BasicBlock *, 4>> LibCallPreds;
  SmallVector<BasicBlock *, 4> InvokePreds;
  bool HasOtherPredTypes = false;
  unsigned NumPredTypes = 0;

  // Partition the predecessors of the landing pad.
  for (BasicBlock *Pred : predecessors(BB)) {
    if (isa<DetachInst>(Pred->getTerminator()) ||
        isa<ReattachInst>(Pred->getTerminator()) ||
        isDetachedRethrow(Pred->getTerminator()))
      DetachPreds.push_back(Pred);
    else if (isTaskFrameResume(Pred->getTerminator()))
      TFResumePreds.push_back(Pred);
    else if (isa<SyncInst>(Pred->getTerminator()))
      SyncPreds.push_back(Pred);
    else if (isSyncUnwind(Pred->getTerminator()))
      SyncUnwindPreds.push_back(Pred);
    else if (CilkSanitizerImpl::isAllocFn(Pred->getTerminator(), TLI))
      AllocFnPreds.push_back(Pred);
    else if (CilkSanitizerImpl::isFreeFn(Pred->getTerminator(), TLI))
      FreeFnPreds.push_back(Pred);
    else if (CilkSanitizerImpl::isLibCall(*Pred->getTerminator(), TLI)) {
      const Function *Called =
          dyn_cast<CallBase>(Pred->getTerminator())->getCalledFunction();
      LibCallPreds[Called].push_back(Pred);
    } else if (isa<InvokeInst>(Pred->getTerminator()))
      InvokePreds.push_back(Pred);
    else
      HasOtherPredTypes = true;
  }

  NumPredTypes = static_cast<unsigned>(!DetachPreds.empty()) +
    static_cast<unsigned>(!TFResumePreds.empty()) +
    static_cast<unsigned>(!SyncPreds.empty()) +
    static_cast<unsigned>(!SyncUnwindPreds.empty()) +
    static_cast<unsigned>(!AllocFnPreds.empty()) +
    static_cast<unsigned>(!FreeFnPreds.empty()) +
    static_cast<unsigned>(LibCallPreds.size()) +
    static_cast<unsigned>(!InvokePreds.empty()) +
    static_cast<unsigned>(HasOtherPredTypes);

  // Splitting predecessors works differently for landingpads versus normal
  // basic blocks.  If the block is not a landingpad, split off every type of
  // predecessor.
  unsigned NumPredTypesRequired = static_cast<unsigned>(BB->isLandingPad());
  if (NumPredTypes <= NumPredTypesRequired)
    return;

  BasicBlock *BBToSplit = BB;
  // Split off the predecessors of each type.
  if (!SyncPreds.empty() && NumPredTypes > NumPredTypesRequired) {
    BBToSplit = splitOffPreds(BBToSplit, SyncPreds, DTU, LI);
    NumPredTypes--;
  }
  if (!SyncUnwindPreds.empty() && NumPredTypes > NumPredTypesRequired) {
    BBToSplit = splitOffPreds(BBToSplit, SyncUnwindPreds, DTU, LI);
    NumPredTypes--;
  }
  if (!AllocFnPreds.empty() && NumPredTypes > NumPredTypesRequired) {
    BBToSplit = splitOffPreds(BBToSplit, AllocFnPreds, DTU, LI);
    NumPredTypes--;
  }
  if (!FreeFnPreds.empty() && NumPredTypes > NumPredTypesRequired) {
    BBToSplit = splitOffPreds(BBToSplit, FreeFnPreds, DTU, LI);
    NumPredTypes--;
  }
  if (!LibCallPreds.empty() && NumPredTypes > NumPredTypesRequired) {
    for (auto KeyVal : LibCallPreds) {
      if (NumPredTypes > NumPredTypesRequired) {
        BBToSplit = splitOffPreds(BBToSplit, KeyVal.second, DTU, LI);
        NumPredTypes--;
      }
    }
  }
  if (!InvokePreds.empty() && NumPredTypes > NumPredTypesRequired) {
    BBToSplit = splitOffPreds(BBToSplit, InvokePreds, DTU, LI);
    NumPredTypes--;
  }
  if (!TFResumePreds.empty() && NumPredTypes > NumPredTypesRequired) {
    BBToSplit = splitOffPreds(BBToSplit, TFResumePreds, DTU, LI);
    NumPredTypes--;
  }
  // We handle detach and detached.rethrow predecessors at the end to preserve
  // invariants on the CFG structure about the deadness of basic blocks after
  // detached-rethrows.
  if (!DetachPreds.empty() && NumPredTypes > NumPredTypesRequired) {
    BBToSplit = splitOffPreds(BBToSplit, DetachPreds, DTU, LI);
    NumPredTypes--;
  }
}

// Setup all basic blocks such that each block's predecessors belong entirely to
// one CSI ID space.
void CilkSanitizerImpl::setupBlocks(Function &F, DominatorTree *DT,
                                    LoopInfo *LI) {
  SmallSetVector<BasicBlock *, 8> BlocksToSetup;
  for (BasicBlock &BB : F) {
    if (BB.isLandingPad())
      BlocksToSetup.insert(&BB);

    if (InvokeInst *II = dyn_cast<InvokeInst>(BB.getTerminator())) {
      if (!isTapirPlaceholderSuccessor(II->getNormalDest()))
        BlocksToSetup.insert(II->getNormalDest());
    } else if (SyncInst *SI = dyn_cast<SyncInst>(BB.getTerminator()))
      BlocksToSetup.insert(SI->getSuccessor(0));
  }

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  for (BasicBlock *BB : BlocksToSetup)
    setupBlock(BB, &DTU, LI, &GetTLI(F));
}

// Do not instrument known races/"benign races" that come from compiler
// instrumentation. The user has no way of suppressing them.
static bool shouldInstrumentReadWriteFromAddress(const Module *M, Value *Addr) {
  // Peel off GEPs and BitCasts.
  Addr = Addr->stripInBoundsOffsets();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->hasSection()) {
      StringRef SectionName = GV->getSection();
      // Check if the global is in the PGO counters section.
      auto OF = Triple(M->getTargetTriple()).getObjectFormat();
      if (SectionName.ends_with(
              getInstrProfSectionName(IPSK_cnts, OF,
                                      /*AddSegmentInfo*/ false)))
        return false;
    }

    // Check if the global is private gcov data.
    if (GV->getName().starts_with("__llvm_gcov") ||
        GV->getName().starts_with("__llvm_gcda"))
      return false;
  }

  // Do not instrument accesses from different address spaces; we cannot deal
  // with them.
  if (Addr) {
    Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
    if (PtrTy->getPointerAddressSpace() != 0)
      return false;
  }

  return true;
}

/// Returns true if Addr can only refer to a locally allocated base object, that
/// is, an object created via an AllocaInst or an AllocationFn.
bool CilkSanitizerImpl::localBaseObj(const Value *Addr, LoopInfo *LI,
                                     const TargetLibraryInfo *TLI) const {
  // If we don't have an address, give up.
  if (!Addr)
    return false;

  // Get the base objects that this address might refer to.
  SmallVectorImpl<const Value *> &BaseObjs = lookupBaseObjects(Addr, LI);

  // If we could not determine the base objects, conservatively return false.
  if (BaseObjs.empty())
    return false;

  // If any base object is not an alloca or allocation function, then it's not
  // local.
  for (const Value *BaseObj : BaseObjs) {
    if (isa<AllocaInst>(BaseObj) || isNoAliasCall(BaseObj))
      continue;

    if (const Argument *A = dyn_cast<Argument>(BaseObj))
      if (A->hasByValAttr())
        continue;

    LLVM_DEBUG(dbgs() << "Non-local base object " << *BaseObj << "\n");
    return false;
  }

  return true;
}

// Examine the uses of a Instruction AI to determine if it is used in a subtask.
// This method assumes that AI is an allocation instruction, i.e., either an
// AllocaInst or an AllocationFn.
bool CilkSanitizerImpl::mightHaveDetachedUse(const Value *V,
                                             const TaskInfo &TI) const {
  // Get the task for this allocation.
  const Task *AllocTask = nullptr;
  if (const Instruction *I = dyn_cast<Instruction>(V))
    AllocTask = TI.getTaskFor(I->getParent());
  else if (const Argument *A = dyn_cast<Argument>(V))
    AllocTask = TI.getTaskFor(&A->getParent()->getEntryBlock());

  // assert(AllocTask && "Null task for instruction.");
  if (!AllocTask) {
    LLVM_DEBUG(dbgs() << "MightHaveDetachedUse: No task found for given value "
               << *V << "\n");
    return false;
  }

  if (AllocTask->isSerial())
    // Alloc AI cannot be used in a subtask if its enclosing task is serial.
    return false;

  SmallVector<const Use *, 20> Worklist;
  SmallSet<const Use *, 20> Visited;

  // Add all uses of AI to the worklist.
  for (const Use &U : V->uses()) {
    Visited.insert(&U);
    Worklist.push_back(&U);
  }

  // Evaluate each use of AI.
  while (!Worklist.empty()) {
    const Use *U = Worklist.pop_back_val();

    // Check if this use of AI is in a different task from the allocation.
    Instruction *I = cast<Instruction>(U->getUser());
    LLVM_DEBUG(dbgs() << "\tExamining use: " << *I << "\n");
    if (AllocTask != TI.getTaskFor(I->getParent())) {
      assert(TI.getTaskFor(I->getParent()) != AllocTask->getParentTask() &&
             "Use of alloca appears in a parent task of that alloca");
      // Because the use of AI cannot appear in a parent task of AI, it must be
      // in a subtask.  In particular, the use cannot be in a shared-EH spindle.
      return true;
    }

    // If the pointer to AI is transformed using one of the following
    // operations, add uses of the transformed pointer to the worklist.
    switch (I->getOpcode()) {
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::PHI:
    case Instruction::Select:
    case Instruction::AddrSpaceCast:
      for (Use &UU : I->uses())
        if (Visited.insert(&UU).second)
          Worklist.push_back(&UU);
      break;
    default:
      break;
    }
  }
  return false;
}

/// Returns true if accesses on Addr could race due to pointer capture.
bool CilkSanitizerImpl::possibleRaceByCapture(const Value *Addr,
                                              const TaskInfo &TI,
                                              LoopInfo *LI) const {
  if (isa<GlobalValue>(Addr))
    // For this analysis, we consider all global values to be captured.
    return true;

  // Check for detached uses of the underlying base objects.
  SmallVectorImpl<const Value *> &BaseObjs = lookupBaseObjects(Addr, LI);

  // If we could not determine the base objects, conservatively return true.
  if (BaseObjs.empty())
    return true;

  for (const Value *BaseObj : BaseObjs) {
    // Skip any null objects
    if (const Constant *C = dyn_cast<Constant>(BaseObj)) {
      // if (C->isNullValue())
      //   continue;
      // Is this value a constant that cannot be derived from any pointer
      // value (we need to exclude constant expressions, for example, that
      // are formed from arithmetic on global symbols).
      bool IsNonPtrConst = isa<ConstantInt>(C) || isa<ConstantFP>(C) ||
                           isa<ConstantPointerNull>(C) ||
                           isa<ConstantDataVector>(C) || isa<UndefValue>(C);
      if (IsNonPtrConst)
        continue;
    }

    // If the base object is not an instruction, conservatively return true.
    if (!isa<Instruction>(BaseObj)) {
      // From BasicAliasAnalysis.cpp: If this is an argument that corresponds to
      // a byval or noalias argument, then it has not escaped before entering
      // the function.
      if (const Argument *A = dyn_cast<Argument>(BaseObj)) {
        if (!A->hasByValAttr() && !A->hasNoAliasAttr())
          return true;
      } else
        return true;
    }

    // If the base object might have a detached use, return true.
    if (lookupMightHaveDetachedUse(BaseObj, TI))
      return true;
  }

  // Perform normal pointer-capture analysis.
  // if (PointerMayBeCaptured(Addr, false, false))
  if (lookupPointerMayBeCaptured(Addr))
    return true;

  return false;
}

bool CilkSanitizerImpl::unknownObjectUses(const Value *Addr, LoopInfo *LI,
                                          const TargetLibraryInfo *TLI) const {
  // Perform normal pointer-capture analysis.
  if (lookupPointerMayBeCaptured(Addr))
    return true;

  // Check for detached uses of the underlying base objects.
  SmallVectorImpl<const Value *> &BaseObjs = lookupBaseObjects(Addr, LI);

  // If we could not determine the base objects, conservatively return true.
  if (BaseObjs.empty())
    return true;

  // If the base object is not an allocation function, return true.
  for (const Value *BaseObj : BaseObjs)
    if (!isAllocFn(BaseObj, TLI))
      return true;

  return false;
}

void CilkSanitizerImpl::chooseInstructionsToInstrument(
    SmallVectorImpl<Instruction *> &Local, SmallVectorImpl<Instruction *> &All,
    const TaskInfo &TI, LoopInfo &LI, const TargetLibraryInfo *TLI) {
  SmallSet<Value*, 8> WriteTargets;
  // Iterate from the end.
  for (Instruction *I : reverse(Local)) {
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      Value *Addr = Store->getPointerOperand();
      if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
        continue;
      WriteTargets.insert(Addr);
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
        continue;
      if (WriteTargets.count(Addr)) {
        // We will write to this temp, so no reason to analyze the read.
        NumOmittedReadsBeforeWrite++;
        continue;
      }
      if (addrPointsToConstantData(Addr)) {
        // Addr points to some constant data -- it can not race with any writes.
        NumOmittedReadsFromConstants++;
        continue;
      }
    }
    Value *Addr = isa<StoreInst>(*I)
        ? cast<StoreInst>(I)->getPointerOperand()
        : cast<LoadInst>(I)->getPointerOperand();
    if (localBaseObj(Addr, &LI, TLI) &&
        !possibleRaceByCapture(Addr, TI, &LI)) {
      // The variable is addressable but not captured, so it cannot be
      // referenced from a different thread and participate in a data race
      // (see llvm/Analysis/CaptureTracking.h for details).
      NumOmittedNonCaptured++;
      continue;
    }
    LLVM_DEBUG(dbgs() << "Pushing " << *I << "\n");
    All.push_back(I);
  }
  Local.clear();
}

bool CilkSanitizerImpl::isLibCall(const Instruction &I,
                                  const TargetLibraryInfo *TLI) {
  if (!isa<CallBase>(I))
    return false;

  if (!TLI)
    return false;

  if (const Function *Called = dyn_cast<CallBase>(&I)->getCalledFunction()) {
    LibFunc F;
    bool FoundLibFunc = TLI->getLibFunc(*Called, F);
    if (FoundLibFunc)
      return true;
  }

  return false;
}

// Helper function to determine if the call-base instruction \p I should be
// skipped when examining calls that affect race detection.  Returns true if and
// only if \p I is a simple call that cannot race.
bool CilkSanitizerImpl::simpleCallCannotRace(const Instruction &I) {
  return callsPlaceholderFunction(I);
}

// Helper function to determine if the call-base instruction \p I should be
// skipped when examining calls that affect race detection.  Returns true if and
// only if \p I is identified as a special function that should be ignored.
bool CilkSanitizerImpl::shouldIgnoreCall(const Instruction &I) {
  if (const CallBase *Call = dyn_cast<CallBase>(&I))
    if (const Function *Called = Call->getCalledFunction())
      if (Called->hasName() && (Called->getName().starts_with("__csi") ||
                                Called->getName().starts_with("__csan") ||
                                Called->getName().starts_with("__cilksan")))
        return true;
  return false;
}

// Helper function to get the ID of a function being called.  These IDs are
// stored in separate global variables in the program.  This method will create
// a new global variable for the Callee's ID if necessary.
Value *CilkSanitizerImpl::getCalleeFuncID(const Function *Callee,
                                          IRBuilder<> &IRB) {
  if (!Callee)
    // Unknown targets (i.e., indirect calls) are always unknown.
    return IRB.getInt64(CsiCallsiteUnknownTargetId);

  std::string GVName =
    CsiFuncIdVariablePrefix + Callee->getName().str();
  GlobalVariable *FuncIdGV = M.getNamedGlobal(GVName);
  Type *FuncIdGVTy = IRB.getInt64Ty();
  if (!FuncIdGV) {
    FuncIdGV =
        dyn_cast<GlobalVariable>(M.getOrInsertGlobal(GVName, FuncIdGVTy));
    assert(FuncIdGV);
    FuncIdGV->setConstant(false);
    if (Options.jitMode && !Callee->empty())
      FuncIdGV->setLinkage(Callee->getLinkage());
    else
      FuncIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
    FuncIdGV->setInitializer(IRB.getInt64(CsiCallsiteUnknownTargetId));
  }
  return IRB.CreateLoad(FuncIdGVTy, FuncIdGV);
}

//------------------------------------------------------------------------------
// SimpleInstrumentor methods, which do not do static race detection.
//------------------------------------------------------------------------------

bool CilkSanitizerImpl::SimpleInstrumentor::instrumentSimpleInstructions(
    SmallVectorImpl<Instruction *> &Instructions) {
  bool Result = false;
  for (Instruction *I : Instructions) {
    bool LocalResult = false;
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      LocalResult |= CilkSanImpl.instrumentLoadOrStore(I);
    else if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
      LocalResult |= CilkSanImpl.instrumentAtomic(I);
    else
      dbgs() << "[Cilksan] Unknown simple instruction: " << *I << "\n";

    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::SimpleInstrumentor::instrumentAnyMemIntrinsics(
    SmallVectorImpl<Instruction *> &MemIntrinsics) {
  bool Result = false;
  for (Instruction *I : MemIntrinsics) {
    bool LocalResult = false;
    if (isa<AnyMemTransferInst>(I)) {
      LocalResult |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, /*Src*/ 1);
      LocalResult |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, /*Dst*/ 0);
    } else {
      assert(isa<AnyMemIntrinsic>(I) &&
             "InstrumentAnyMemIntrinsics operating on not a memory intrinsic.");
      LocalResult |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, unsigned(-1));
    }
    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::SimpleInstrumentor::instrumentCalls(
    SmallVectorImpl<Instruction *> &Calls) {
  bool Result = false;
  for (Instruction *I : Calls) {
    // Allocation-function and free calls are handled separately.
    if (isAllocFn(I, TLI) || isFreeFn(I, TLI))
      continue;

    bool LocalResult = false;
    if (isa<IntrinsicInst>(I))
      LocalResult |=
          CilkSanImpl.instrumentIntrinsicCall(I, /*MAAPVals*/ nullptr);
    else if (isLibCall(*I, TLI))
      LocalResult |=
          CilkSanImpl.instrumentLibCall(I, /*MAAPVals*/ nullptr);
    else
      LocalResult |= CilkSanImpl.instrumentCallsite(I, /*MAAPVals*/ nullptr);
    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::SimpleInstrumentor::instrumentAncillaryInstructions(
    SmallPtrSetImpl<Instruction *> &Allocas,
    SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
    SmallPtrSetImpl<Instruction *> &FreeCalls,
    DenseMap<Value *, unsigned> &SyncRegNums,
    DenseMap<BasicBlock *, unsigned> &SRCounters, const DataLayout &DL) {
  bool Result = false;
  SmallSetVector<SyncInst *, 4> Syncs;
  SmallSetVector<Loop *, 4> Loops;

  // Instrument allocas and allocation-function calls that may be involved in a
  // race.
  for (Instruction *I : inProgramOrder(Allocas)) {
    // The simple instrumentor just instruments everything
    CilkSanImpl.instrumentAlloca(I, TI);
    getDetachesForInstruction(I);
    Result = true;
  }
  for (Instruction *I : inProgramOrder(AllocationFnCalls)) {
    // The simple instrumentor just instruments everything
    CilkSanImpl.instrumentAllocationFn(I, DT, TLI);
    getDetachesForInstruction(I);
    Result = true;
  }
  for (Instruction *I : inProgramOrder(FreeCalls)) {
    // The first argument of the free call is the pointer.
    Value *Ptr = I->getOperand(0);
    // If the pointer corresponds to an allocation function call in this
    // function, then instrument it.
    if (Instruction *PtrI = dyn_cast<Instruction>(Ptr)) {
      if (AllocationFnCalls.count(PtrI)) {
        CilkSanImpl.instrumentFree(I, TLI);
        getDetachesForInstruction(I);
        Result = true;
        continue;
      }
    }
    // The simple instrumentor just instruments everything
    CilkSanImpl.instrumentFree(I, TLI);
    getDetachesForInstruction(I);
    Result = true;
  }

  // Instrument detaches
  for (DetachInst *DI : Detaches) {
    CilkSanImpl.instrumentDetach(DI, SyncRegNums[DI->getSyncRegion()],
                                 SRCounters[DI->getDetached()], DT, TI, LI);
    Result = true;
    // Get syncs associated with this detach
    for (SyncInst *SI : CilkSanImpl.DetachToSync[DI])
      Syncs.insert(SI);

    if (CilkSanImpl.Options.InstrumentLoops) {
      // Get any loop associated with this detach.
      Loop *L = LI.getLoopFor(DI->getParent());
      if (spawnsTapirLoopBody(DI, LI, TI))
        Loops.insert(L);
    }
  }

  // Instrument associated syncs
  for (SyncInst *SI : Syncs)
    CilkSanImpl.instrumentSync(SI, SyncRegNums[SI->getSyncRegion()]);

  if (CilkSanImpl.Options.InstrumentLoops) {
    // Recursively instrument all Tapir loops
    for (Loop *L : Loops)
      CilkSanImpl.instrumentTapirLoop(*L, TI, SyncRegNums);
  }

  return Result;
}

// TODO: Combine this redundant logic with that in Instrumentor
void CilkSanitizerImpl::SimpleInstrumentor::getDetachesForInstruction(
    Instruction *I) {
  // Get the Task for I.
  Task *T = TI.getTaskFor(I->getParent());
  // Add the ancestors of T to the set of detaches to instrument.
  while (!T->isRootTask()) {
    // Once we encounter a detach we've previously added to the set, we know
    // that all its parents are also in the set.
    if (!Detaches.insert(T->getDetach()))
      return;
    T = T->getParentTask();
  }
}

//------------------------------------------------------------------------------
// Instrumentor methods
//------------------------------------------------------------------------------

void CilkSanitizerImpl::Instrumentor::getDetachesForInstruction(
    Instruction *I) {
  // Get the Task for I.
  Task *T = TI.getTaskFor(I->getParent());
  // Add the ancestors of T to the set of detaches to instrument.
  while (!T->isRootTask()) {
    // Once we encounter a detach we've previously added to the set, we know
    // that all its parents are also in the set.
    if (!Detaches.insert(T->getDetach()))
      return;
    T = T->getParentTask();
  }
}

unsigned CilkSanitizerImpl::Instrumentor::raceTypeToFlagVal(
    RaceInfo::RaceType RT) {
  unsigned FlagVal = static_cast<unsigned>(MAAPValue::NoAccess);
  if (RaceInfo::isLocalRace(RT) || RaceInfo::isOpaqueRace(RT))
    FlagVal = static_cast<unsigned>(MAAPValue::ModRef);
  if (RaceInfo::isRaceViaAncestorMod(RT))
    FlagVal |= static_cast<unsigned>(MAAPValue::Mod);
  if (RaceInfo::isRaceViaAncestorRef(RT))
    FlagVal |= static_cast<unsigned>(MAAPValue::Ref);
  return FlagVal;
}

static Value *getMAAPIRValue(IRBuilder<> &IRB, unsigned MV) {
  return IRB.getInt8(MV);
}

// Insert per-argument MAAPs for this function
void CilkSanitizerImpl::Instrumentor::insertArgMAAPs(Function &F,
                                                     Value *FuncId) {
  if (!MAAPChecks)
    return;
  LLVM_DEBUG(dbgs() << "InsertArgMAAPs: " << F.getName() << "\n");
  IRBuilder<> IRB(cast<Instruction>(FuncId)->getNextNode());
  unsigned ArgIdx = 0;
  for (Argument &Arg : F.args()) {
    if (!Arg.getType()->isPtrOrPtrVectorTy())
      continue;

    // Create a new flag for this argument MAAP.
    Type *MAAPIRValueTy = getMAAPIRValue(IRB, 0)->getType();
    Value *NewFlag = IRB.CreateAlloca(MAAPIRValueTy,
                                      Arg.getType()->getPointerAddressSpace());
    Value *FinalMV;
    // If this function is main, then it has no ancestors that can create races.
    if (F.getName() == "main") {
      FinalMV = getMAAPIRValue(IRB, raceTypeToFlagVal(RaceInfo::None));
      IRB.CreateStore(FinalMV, NewFlag);
    } else {
      // Call the runtime function to set the value of this flag.
      IRB.CreateCall(CilkSanImpl.GetMAAP,
                     {NewFlag, FuncId, IRB.getInt8(ArgIdx)});

      // Incorporate local information into this MAAP value.
      unsigned LocalMV = static_cast<unsigned>(MAAPValue::NoAccess);
      if (Arg.hasNoAliasAttr())
        LocalMV |= static_cast<unsigned>(MAAPValue::NoAlias);

      // Store this local MAAP value.
      FinalMV = IRB.CreateOr(getMAAPIRValue(IRB, LocalMV),
                             IRB.CreateLoad(MAAPIRValueTy, NewFlag));
      IRB.CreateStore(FinalMV, NewFlag);
    }
    // Associate this flag with the argument for future lookups.
    LLVM_DEBUG(dbgs() << "Recording local MAAP for arg " << Arg << ": "
                      << *NewFlag << "\n");
    LocalMAAPs[&Arg] = FinalMV;
    ArgMAAPs.insert(FinalMV);
    ++ArgIdx;
  }

  // Record other objects known to be involved in races.
  for (auto &ObjRD : RI.getObjectMRForRace()) {
    if (isa<Instruction>(ObjRD.first)) {
      unsigned MAAPVal = static_cast<unsigned>(MAAPValue::NoAccess);
      if (isModSet(ObjRD.second))
        MAAPVal |= static_cast<unsigned>(MAAPValue::Mod);
      if (isRefSet(ObjRD.second))
        MAAPVal |= static_cast<unsigned>(MAAPValue::Ref);
      // Determine if this object is no-alias.
      if (const CallBase *CB = dyn_cast<CallBase>(ObjRD.first)) {
        if (CB->hasRetAttr(Attribute::NoAlias))
          MAAPVal |= static_cast<unsigned>(MAAPValue::NoAlias);
      } else if (isa<AllocaInst>(ObjRD.first))
        MAAPVal |= static_cast<unsigned>(MAAPValue::NoAlias);

      LLVM_DEBUG(dbgs() << "Setting LocalMAAPs for " << *ObjRD.first << " = "
                        << MAAPVal << "\n");
      LocalMAAPs[ObjRD.first] = getMAAPIRValue(IRB, MAAPVal);
    }
  }
}

bool CilkSanitizerImpl::Instrumentor::instrumentSimpleInstructions(
    SmallVectorImpl<Instruction *> &Instructions) {
  bool Result = false;
  for (Instruction *I : Instructions) {
    bool LocalResult = false;
    // Simple instructions, such as loads, stores, or atomics, have just one
    // pointer operand, and therefore should have at most one entry of RaceData.

    // If the instruction might participate in a local or opaque race,
    // instrument it unconditionally.
    if (RI.mightRaceOpaquely(I)) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I))
        LocalResult |= CilkSanImpl.instrumentLoadOrStore(I);
      else if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
        LocalResult |= CilkSanImpl.instrumentAtomic(I);
      else
        dbgs() << "[Cilksan] Unknown simple instruction: " << *I << "\n";
    } else if (RI.mightRaceViaAncestor(I) || RI.mightRaceLocally(I)) {
      // Otherwise, if the instruction might participate in a race via an
      // ancestor function instantiation, instrument it conditionally, based on
      // the pointer.
      //
      // Delay handling this instruction.
      DelayedSimpleInsts.push_back(I);
      LocalResult |= true;
    }

    // If any instrumentation was inserted, collect associated instructions to
    // instrument.
    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::Instrumentor::instrumentAnyMemIntrinsics(
    SmallVectorImpl<Instruction *> &MemIntrinsics) {
  bool Result = false;
  for (Instruction *I : MemIntrinsics) {
    bool LocalResult = false;
    // If this instruction cannot race, skip it.
    if (!RI.mightRace(I))
      continue;

    // Look over the race data to determine what memory intrinsics need to be
    // instrumented and how.
    SmallSet<std::pair<Instruction *, unsigned>, 2> ToInstrument;
    SmallSet<std::pair<Instruction *, unsigned>, 2> MaybeDelay;
    for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
      assert(RD.getPtr() && "No pointer for race with memory intrinsic.");
      if (RaceInfo::isOpaqueRace(RD.Type)) {
        ToInstrument.insert(std::make_pair(I, RD.OperandNum));
        LocalResult |= true;
      } else if (RaceInfo::isRaceViaAncestor(RD.Type) ||
                 RaceInfo::isLocalRace(RD.Type)) {
        // Possibly delay handling this instruction.
        MaybeDelay.insert(std::make_pair(I, RD.OperandNum));
        LocalResult |= true;
      }
    }

    // Do the instrumentation
    for (const std::pair<Instruction *, unsigned> &MemIntrin : ToInstrument)
      CilkSanImpl.instrumentAnyMemIntrinAcc(MemIntrin.first, MemIntrin.second);
    for (const std::pair<Instruction *, unsigned> &MemIntrin : MaybeDelay)
      if (!ToInstrument.count(MemIntrin))
        DelayedMemIntrinsics.push_back(MemIntrin);

    // If any instrumentation was inserted, collect associated instructions to
    // instrument.
    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::Instrumentor::instrumentCalls(
    SmallVectorImpl<Instruction *> &Calls) {
  bool Result = false;
  for (Instruction *I : Calls) {
    // Allocation-function and free calls are handled separately.
    if (isAllocFn(I, TLI) || isFreeFn(I, TLI))
      continue;

    bool LocalResult = false;
    bool GetDetaches = false;

    // Get current race data for this call.
    RaceInfo::RaceType CallRT = RI.getRaceType(I);
    LLVM_DEBUG({
        dbgs() << "Call " << *I << ": ";
        RaceInfo::printRaceType(CallRT, dbgs());
        dbgs() << "\n";
      });

    // Get update race data, if it's available.
    RaceInfo::RaceType FuncRT = CallRT;
    CallBase *CB = dyn_cast<CallBase>(I);
    if (Function *CF = CB->getCalledFunction())
      if (CilkSanImpl.FunctionRaceType.count(CF))
        FuncRT = CilkSanImpl.FunctionRaceType[CF];

    LLVM_DEBUG({
        dbgs() << "  FuncRT: ";
        RaceInfo::printRaceType(FuncRT, dbgs());
        dbgs() << "\n";
      });

    // Propagate information about opaque races from function to call.
    if (!RaceInfo::isOpaqueRace(FuncRT))
      CallRT = RaceInfo::clearOpaqueRace(CallRT);

    LLVM_DEBUG({
        dbgs() << "  New CallRT: ";
        RaceInfo::printRaceType(CallRT, dbgs());
        dbgs() << "\n";
      });

    // If this instruction cannot race, see if we can suppress it
    if (!RaceInfo::isRace(CallRT)) {
      // Nothing to suppress if this is an intrinsic
      if (isa<IntrinsicInst>(I))
        continue;

      // We can only suppress calls whose functions don't have local races.
      if (!RaceInfo::isLocalRace(FuncRT)) {
        if (!CB->doesNotAccessMemory())
          LocalResult |= CilkSanImpl.suppressCallsite(I);
        continue;
      // } else {
      //   GetDetaches |= CilkSanImpl.instrumentCallsite(I);
      //   // SmallPtrSet<Value *, 1> Objects;
      //   // RI.getObjectsFor(I, Objects);
      //   // for (Value *Obj : Objects) {
      //   //   CilkSanImpl.ObjectMRForRace[Obj] = ModRefInfo::ModRef;
      //   // }
      }
      // continue;
    }

    // We're going to instrument this call for potential races.  First get
    // MAAP information for its arguments, if any races depend on the
    // ancestor.
    SmallVector<Value *, 8> MAAPVals;
    LLVM_DEBUG(dbgs() << "Getting MAAP values for " << *CB << "\n");
    IRBuilder<> IRB(I);
    unsigned OpIdx = 0;
    for (const Value *Op : CB->args()) {
      if (!MAAPChecks)
        continue;

      if (!Op->getType()->isPtrOrPtrVectorTy()) {
        ++OpIdx;
        continue;
      }

      // Check if this operand might race via ancestor.
      bool RaceViaAncestor = false;
      for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
        if (RD.OperandNum != OpIdx)
          continue;
        if (RaceInfo::isRaceViaAncestor(RD.Type)) {
          RaceViaAncestor = true;
          break;
        }
      }

      Value *MAAPVal;
      if (RaceViaAncestor)
        // Evaluate race data for I and OpIdx to compute the MAAP value.
        MAAPVal = getMAAPValue(I, IRB, OpIdx);
      else
        // We have either an opaque race or a local race, but _not_ a race via
        // an ancestor.  We want to propagate MAAP information on pointer
        // arguments, but we don't need to be pessimistic when a value can't be
        // found.
        MAAPVal = getMAAPValue(I, IRB, OpIdx, MAAPValue::NoAccess,
                               /*CheckArgs*/ false);
      LLVM_DEBUG({
          dbgs() << "  Op: " << *CB->getArgOperand(OpIdx) << "\n";
          dbgs() << "  MAAP value: " << *MAAPVal << "\n";
        });
      MAAPVals.push_back(MAAPVal);
      ++OpIdx;
    }

    Value *CalleeID = CilkSanImpl.getCalleeFuncID(CB->getCalledFunction(), IRB);
    // We set the MAAPs in reverse order to support stack-like access of the
    // MAAPs by in-order calls to GetMAAP in the callee.
    for (Value *MAAPVal : reverse(MAAPVals))
      IRB.CreateCall(CilkSanImpl.SetMAAP, {MAAPVal, CalleeID});

    if (isa<IntrinsicInst>(I))
      GetDetaches |= CilkSanImpl.instrumentIntrinsicCall(I, &MAAPVals);
    else if (isLibCall(*I, TLI))
      GetDetaches |= CilkSanImpl.instrumentLibCall(I, &MAAPVals);
    else
      GetDetaches |= CilkSanImpl.instrumentCallsite(I, &MAAPVals);

    // If any instrumentation was inserted, collect associated instructions to
    // instrument.
    Result |= LocalResult;
    if (GetDetaches) {
      Result |= GetDetaches;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

Value *CilkSanitizerImpl::Instrumentor::readMAAPVal(Value *V,
                                                    IRBuilder<> &IRB) {
  if (!ArgMAAPs.count(V))
    return V;
  // Marking the load as invariant is not technically correct, because the
  // __csan_get_MAAP call sets the value.  But this call happens
  // once, and all subsequent loads will return the same value.
  //
  // MDNode *MD = llvm::MDNode::get(IRB.getContext(), llvm::None);
  // cast<Instruction>(Load)->setMetadata(LLVMContext::MD_invariant_load, MD);

  // TODO: See if there's a better way to annotate this load for optimization.
  // LoadInst *I = IRB.CreateLoad(V);
  // if (auto *IMD = I->getMetadata(LLVMContext::MD_invariant_group))
  //   I->setMetadata(LLVMContext::MD_invariant_group, IMD);
  // else
  //   I->setMetadata(LLVMContext::MD_invariant_group,
  //                  MDNode::get(IRB.getContext(), {}));
  Value *MV;
  if (AllocaInst *A = dyn_cast<AllocaInst>(V))
    MV = IRB.CreateLoad(A->getAllocatedType(), A);
  else
    MV = V;
  return MV;
}

// Get the memory location for this instruction and operand.
static MemoryLocation getMemoryLocation(Instruction *I, unsigned OperandNum,
                                        const TargetLibraryInfo *TLI) {
  if (auto *MI = dyn_cast<AnyMemIntrinsic>(I)) {
    if (auto *MT = dyn_cast<AnyMemTransferInst>(I)) {
      if (OperandNum == 1)
        return MemoryLocation::getForSource(MT);
    }
    return MemoryLocation::getForDest(MI);
  }
  if (OperandNum == static_cast<unsigned>(-1)) {
    return MemoryLocation::get(I);
  }
  assert(isa<CallBase>(I) &&
         "Unknown instruction and operand ID for getting MemoryLocation.");
  CallBase *CB = cast<CallBase>(I);
  return MemoryLocation::getForArgument(CB, OperandNum, TLI);
}

// Evaluate the noalias value in the MAAP for Obj, and intersect that result
// with the noalias information for other objects.
// The objects in S in an order that doesn't depend on their addresses, as
// SmallPtrSet's does: arguments by number, then instructions in function
// order, then globals in module order. Iterating S directly would make the
// MAAP code's output, and so the instrumented IR, vary from run to run.
static SmallVector<const Value *, 4>
stableObjectOrder(const SmallPtrSetImpl<const Value *> &S) {
  SmallVector<const Value *, 4> V(S.begin(), S.end());
  if (V.size() < 2)
    return V;
  DenseMap<const BasicBlock *, unsigned> BlockNum;
  DenseMap<const GlobalValue *, unsigned> GlobalNum;
  for (const Value *X : V) {
    if (auto *I = dyn_cast<Instruction>(X); I && BlockNum.empty()) {
      unsigned N = 0;
      for (const BasicBlock &BB : *I->getFunction())
        BlockNum[&BB] = N++;
    } else if (auto *G = dyn_cast<GlobalValue>(X); G && GlobalNum.empty()) {
      unsigned N = 0;
      for (const GlobalValue &H : G->getParent()->global_values())
        GlobalNum[&H] = N++;
    }
  }
  auto Rank = [](const Value *X) {
    return isa<Argument>(X) ? 0 : isa<Instruction>(X) ? 1
                              : isa<GlobalValue>(X)   ? 2
                                                      : 3;
  };
  llvm::stable_sort(V, [&](const Value *A, const Value *B) {
    if (Rank(A) != Rank(B))
      return Rank(A) < Rank(B);
    if (auto *X = dyn_cast<Argument>(A))
      return X->getArgNo() < cast<Argument>(B)->getArgNo();
    if (auto *X = dyn_cast<Instruction>(A)) {
      auto *Y = cast<Instruction>(B);
      if (X->getParent() != Y->getParent())
        return BlockNum.lookup(X->getParent()) <
               BlockNum.lookup(Y->getParent());
      return X->comesBefore(Y);
    }
    if (auto *X = dyn_cast<GlobalValue>(A))
      return GlobalNum.lookup(X) < GlobalNum.lookup(cast<GlobalValue>(B));
    return false;
  });
  return V;
}

Value *CilkSanitizerImpl::Instrumentor::getNoAliasMAAPValue(
    Instruction *I, IRBuilder<> &IRB, unsigned OperandNum,
    MemoryLocation Loc, const RaceInfo::RaceData &RD, const Value *Obj,
    Value *ObjNoAliasFlag) {
  AAResults *AA = RI.getAA();

  for (const RaceInfo::RaceData &OtherRD : RI.getRaceData(I)) {
    // Skip checking other accesses that don't involve a pointer
    if (!OtherRD.Access.getPointer())
      continue;
    // Skip this operand when scanning for aliases
    if (OperandNum == OtherRD.OperandNum)
      continue;

    // If we can tell statically that these two memory locations don't alias,
    // move on.
    if (!AA->alias(Loc, getMemoryLocation(I, OtherRD.OperandNum, TLI)))
      continue;

    // We trust that the MAAP value in LocalMAAPs[] for this object Obj, set by
    // InsertArgMAAPs, is correct.  We need to check the underlying objects of
    // the other arguments to see if they match this object.

    // Otherwise we check the underlying objects.
    SmallPtrSet<const Value *, 1> OtherObjects;
    RI.getObjectsFor(OtherRD.Access, OtherObjects);
    for (const Value *OtherObj : stableObjectOrder(OtherObjects)) {
      // If we find another instance of this object in another argument,
      // then we don't have "no alias".
      if (Obj == OtherObj) {
        LLVM_DEBUG({
            dbgs() << "getNoAliasMAAPValue: Matching objects found:\n";
            dbgs() << "  Obj: " << *Obj << "\n";
            dbgs() << "    I: " << *I << "\n";
            dbgs() << " Operands " << OperandNum << ", " << OtherRD.OperandNum
                   << "\n";
          });
        return getMAAPIRValue(IRB, 0);
      }

      // We now know that Obj and OtherObj don't match.

      // If the other object is an argument, then we trust the noalias value in
      // the MAAP for Obj.
      if (isa<Argument>(OtherObj))
        continue;

      // // If the other object is something we can't reason about locally, then we
      // // give up.
      // if (!isa<Instruction>(OtherObj))
      //   return getMAAPIRValue(IRB, 0);

      // Otherwise, check if the other object might alias this one.
      if (AA->alias(Loc, MemoryLocation::getBeforeOrAfter(OtherObj))) {
        LLVM_DEBUG({
            dbgs() << "getNoAliasMAAPValue: Possible aliasing between:\n";
            dbgs() << "  Obj: " << *Obj << "\n";
            dbgs() << "  OtherObj: " << *OtherObj << "\n";
          });
        return getMAAPIRValue(IRB, 0);
      }
    }
  }
  return ObjNoAliasFlag;
}

Value *CilkSanitizerImpl::Instrumentor::getMAAPValue(Instruction *I,
                                                     IRBuilder<> &IRB,
                                                     unsigned OperandNum,
                                                     MAAPValue DefaultMV,
                                                     bool CheckArgs) {
  Function *F = I->getFunction();
  AAResults *AA = RI.getAA();
  MemoryLocation Loc = getMemoryLocation(I, OperandNum, TLI);
  Value *MV = getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAccess));
  Value *DefaultMAAP = getMAAPIRValue(IRB, static_cast<unsigned>(DefaultMV));
  Value *NoAliasFlag =
      getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAlias));

  // If I is a call, check if any other arguments of this call alias the
  // specified operand.
  if (const CallBase *CB = dyn_cast<CallBase>(I)) {
    unsigned OpIdx = 0;
    bool FoundAliasingArg = false;
    for (const Value *Arg : CB->args()) {
      // Skip this operand and any operands that are not pointers.
      // TODO: Handle pointer-vector types.
      if (OpIdx == OperandNum || !Arg->getType()->isPointerTy()) {
        ++OpIdx;
        continue;
      }

      // If this argument does not alias Loc, skip it.
      if (!AA->alias(Loc, getMemoryLocation(I, OpIdx, TLI))) {
        ++OpIdx;
        continue;
      }

      // If the operands must alias, then discard the default noalias MAAP
      // value.
      AliasResult ArgAlias = AA->alias(Loc, getMemoryLocation(I, OpIdx, TLI));
      if (AliasResult::MustAlias == ArgAlias ||
          AliasResult::PartialAlias == ArgAlias) {
        NoAliasFlag = getMAAPIRValue(IRB, 0);
        break;
      }

      // Get objects corresponding to this argument.
      SmallPtrSet<const Value *, 1> ArgObjects;
      RI.getObjectsFor(RaceInfo::MemAccessInfo(
                           Arg, isModSet(AA->getArgModRefInfo(CB, OpIdx))),
                       ArgObjects);
      for (const Value *Obj : stableObjectOrder(ArgObjects)) {
        // If Loc and the racer object cannot alias, then there's nothing to
        // check.
        if (!AA->alias(Loc, MemoryLocation::getBeforeOrAfter(Obj)))
          continue;

        // If we have no local MAAP data for Obj, then act pessimally.
        if (!LocalMAAPs.count(Obj)) {
          FoundAliasingArg = true;
          break;
        }

        // Intersect the dynamic noalias information for this object into the
        // noalias flag.
        Value *FlagLoad = readMAAPVal(LocalMAAPs[Obj], IRB);
        Value *ObjNoAliasFlag = IRB.CreateAnd(
            FlagLoad,
            getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAlias)));
        NoAliasFlag = IRB.CreateAnd(NoAliasFlag, ObjNoAliasFlag);
      }

      if (FoundAliasingArg) {
        // If we found an aliasing argument, fall back to noalias = false.
        NoAliasFlag = getMAAPIRValue(IRB, 0);
        break;
      }
      ++OpIdx;
    }
  }

  // Check the recorded race data for I.
  for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
    // Skip race data for different operands of the same instruction.
    if (OperandNum != RD.OperandNum)
      continue;

    // Otherwise use information about the possibly accessed objects to
    // determine the MAAP value.
    SmallPtrSet<const Value *, 1> Objects;
    RI.getObjectsFor(RD.Access, Objects);

    // If we have a valid racer, get the objects that that racer might access.
    SmallPtrSet<const Value *, 1> RacerObjects;
    unsigned LocalRaceVal = static_cast<unsigned>(MAAPValue::NoAccess);
    if (RD.Racer.isValid()) {
      // Get the local race value for this racer
      assert(RaceInfo::isLocalRace(RD.Type) && "Valid racer for nonlocal race");
      RI.getObjectsFor(
          RaceInfo::MemAccessInfo(RD.Racer.getPtr(), RD.Racer.isMod()),
          RacerObjects);
      if (RD.Racer.isMod())
        LocalRaceVal |= static_cast<unsigned>(MAAPValue::Mod);
      if (RD.Racer.isRef())
        LocalRaceVal |= static_cast<unsigned>(MAAPValue::Ref);
    }

    // Get MAAPs from objects
    for (const Value *Obj : stableObjectOrder(Objects)) {
      // If we find an object with no MAAP, give up.
      if (!LocalMAAPs.count(Obj)) {
        LLVM_DEBUG(dbgs() << "No local MAAP found for obj " << *Obj << "\n");
        if (RD.Racer.isValid())
          MV = IRB.CreateOr(MV, getMAAPIRValue(IRB, LocalRaceVal));
        else
          MV = IRB.CreateOr(MV, DefaultMAAP);
        continue;
      }

      Value *FlagLoad = readMAAPVal(LocalMAAPs[Obj], IRB);
      Value *FlagCheck = IRB.CreateAnd(
          FlagLoad, getMAAPIRValue(IRB, raceTypeToFlagVal(RD.Type)));
      MV = IRB.CreateOr(MV, FlagCheck);

      // Get the dynamic no-alias bit from the MAAP value.
      Value *ObjNoAliasFlag = IRB.CreateAnd(
          FlagLoad,
          getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAlias)));
      Value *NoAliasCheck =
          IRB.CreateICmpNE(getMAAPIRValue(IRB, 0), ObjNoAliasFlag);

      if (RD.Racer.isValid()) {
        for (const Value *RObj : stableObjectOrder(RacerObjects)) {
          // If the racer object matches Obj, there's no need to check a flag.
          if (RObj == Obj) {
            MV = IRB.CreateOr(MV, LocalRaceVal);
            continue;
          }

          // If Loc and the racer object cannot alias, then there's nothing to
          // check.
          if (!AA->alias(Loc, MemoryLocation::getBeforeOrAfter(RObj)))
            continue;

          // If there is must or partial aliasing between this object and racer
          // object, or we have no local MAAP information for RObj, then
          // act conservatively, because there's nothing to check.
          if (AliasResult::MustAlias ==
                  AA->alias(Loc, MemoryLocation::getBeforeOrAfter(RObj)) ||
              AliasResult::PartialAlias ==
                  AA->alias(Loc, MemoryLocation::getBeforeOrAfter(RObj)) ||
              !LocalMAAPs.count(RObj)) {
            if (!LocalMAAPs.count(RObj))
              LLVM_DEBUG(dbgs() << "No local MAAP found for racer object "
                                << *RObj << "\n");
            else
              LLVM_DEBUG(dbgs() << "AA indicates must or partial alias with "
                                   "racer object "
                                << *RObj << "\n");
            MV = IRB.CreateOr(MV, LocalRaceVal);
            continue;
          }

          // These two objects may alias, based on static analysis.  Check the
          // dynamic MAAP values.  We can suppress the race if either this
          // object or the racer object is dynamically noalias, i.e., if either
          // was derived from an allocation or noalias function argument.
          Value *FlagLoad = readMAAPVal(LocalMAAPs[RObj], IRB);
          Value *RObjNoAliasFlag = IRB.CreateAnd(
              FlagLoad,
              getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAlias)));
          Value *RObjNoAliasCheck =
              IRB.CreateICmpNE(getMAAPIRValue(IRB, 0), RObjNoAliasFlag);
          Value *FlagCheck = IRB.CreateSelect(
              IRB.CreateOr(NoAliasCheck, RObjNoAliasCheck),
              getMAAPIRValue(IRB, 0),
              IRB.CreateAnd(FlagLoad, getMAAPIRValue(IRB, LocalRaceVal)));
          MV = IRB.CreateOr(MV, FlagCheck);
        }
      } else if (CheckArgs) {
        // Check the function arguments that might alias this object.
        for (Argument &Arg : F->args()) {
          // Ignore non-pointer arguments
          if (!Arg.getType()->isPtrOrPtrVectorTy())
            continue;
          // Ignore any arguments that match checked objects.
          if (&Arg == Obj)
            continue;
          // Check if Loc and Arg may alias.
          if (!AA->alias(Loc, MemoryLocation::getBeforeOrAfter(&Arg)))
            continue;
          // If we have no local MAAP information about the argument,
          // then there's nothing to check.
          if (!LocalMAAPs.count(&Arg)) {
            LLVM_DEBUG(dbgs() << "No local MAAP found for arg " << Arg << "\n");
            MV = IRB.CreateOr(MV, DefaultMAAP);
            continue;
          }

          // These two objects may alias, based on static analysis.  Check the
          // dynamic MAAP values.  We can suppress the race if either
          // this object or the racer object is dynamically noalias, i.e., if
          // either was derived from an allocation or noalias function argument.
          Value *FlagLoad = readMAAPVal(LocalMAAPs[&Arg], IRB);
          Value *ArgNoAliasFlag = IRB.CreateAnd(
              FlagLoad,
              getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAlias)));
          Value *ArgNoAliasCheck =
              IRB.CreateICmpNE(getMAAPIRValue(IRB, 0), ArgNoAliasFlag);
          Value *FlagCheck = IRB.CreateSelect(
              IRB.CreateOr(NoAliasCheck, ArgNoAliasCheck),
              getMAAPIRValue(IRB, 0),
              IRB.CreateAnd(FlagLoad,
                            getMAAPIRValue(IRB, raceTypeToFlagVal(RD.Type))));
          MV = IRB.CreateOr(MV, FlagCheck);
        }
      }
      // Call getNoAliasMAAPValue to evaluate the no-alias value in the
      // MAAP for Obj, and intersect that result with the noalias
      // information for other objects.
      NoAliasFlag = IRB.CreateAnd(NoAliasFlag,
                                  getNoAliasMAAPValue(I, IRB, OperandNum, Loc,
                                                      RD, Obj, ObjNoAliasFlag));
    }
  }
  // Record the no-alias information.
  MV = IRB.CreateOr(MV, NoAliasFlag);
  return MV;
}

Value *CilkSanitizerImpl::Instrumentor::getMAAPCheck(Instruction *I,
                                                     IRBuilder<> &IRB,
                                                     unsigned OperandNum) {
  Function *F = I->getFunction();
  bool LocalRace = RI.mightRaceLocally(I);
  AAResults *AA = RI.getAA();
  MemoryLocation Loc = getMemoryLocation(I, OperandNum, TLI);
  Value *MAAPChk = IRB.getTrue();
  if (LocalRace)
    return IRB.getFalse();

  // Check the recorded race data for I.
  for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
    LLVM_DEBUG(dbgs() << "  Race Data:\n  Ptr = " << *RD.getPtr() << "\n");
    LLVM_DEBUG(RaceInfo::printRaceType(RD.Type, dbgs() << "  "));
    LLVM_DEBUG(dbgs() << "\n");
    LLVM_DEBUG(dbgs() << "    current MAAPChk = " << *MAAPChk << "\n");
    // Skip race data for different operands of the same instruction.
    if (OperandNum != RD.OperandNum)
      continue;

    // If this racer is opaque, then we can't create a valid MAAP check for it.
    if (RaceInfo::isOpaqueRace(RD.Type))
      return IRB.getFalse();

    // If this racer is local, then skip it.  We've already accommodated local
    // races via runtime pointer checks, if available.
    if (RaceInfo::isLocalRace(RD.Type))
      continue;

    LLVM_DEBUG(dbgs() << "  Getting objects for racer\n");

    SmallPtrSet<const Value *, 1> Objects;
    RI.getObjectsFor(RD.Access, Objects);

    // If we have a valid racer, get the objects that that racer might access.
    SmallPtrSet<const Value *, 1> RacerObjects;

    for (const Value *Obj : stableObjectOrder(Objects)) {
      LLVM_DEBUG(dbgs() << "  Object " << *Obj << "\n");
      LLVM_DEBUG(dbgs() << "    current MAAPChk = " << *MAAPChk << "\n");
      // Ignore objects that are not involved in races.
      if (!RI.ObjectInvolvedInRace(Obj))
        continue;

      // If we find an object with no MAAP, give up.
      if (!LocalMAAPs.count(Obj)) {
        LLVM_DEBUG(dbgs() << "No local MAAP found for object " << *Obj << "\n"
                          << "  I: " << *I << "\n"
                          << "  Ptr: " << *RD.Access.getPointer() << "\n");
        return IRB.getFalse();
      }

      Value *FlagLoad = readMAAPVal(LocalMAAPs[Obj], IRB);
      LLVM_DEBUG(dbgs() << "  FlagLoad " << *FlagLoad << "\n");

      // If we're dealing with a local race, then don't suppress based on the
      // race-type information from the MAAP value.  For function arguments,
      // that MAAP value reflects potential races via an ancestor, which should
      // not disable checking of local races.
      Value *LocalCheck;
      Value *FlagCheck = IRB.CreateAnd(
          FlagLoad, getMAAPIRValue(IRB, raceTypeToFlagVal(RD.Type)));
      LLVM_DEBUG(dbgs() << "  FlagCheck " << *FlagCheck << "\n");
      LocalCheck = IRB.CreateICmpEQ(getMAAPIRValue(IRB, 0), FlagCheck);
      LLVM_DEBUG(dbgs() << "  LocalCheck " << *LocalCheck << "\n");

      // Add the check.
      MAAPChk = IRB.CreateAnd(MAAPChk, LocalCheck);
      LLVM_DEBUG(dbgs() << "  MAAPChk " << *MAAPChk << "\n");

      // Get the dynamic no-alias bit from the MAAP value.
      Value *NoAliasCheck = IRB.CreateICmpNE(
          getMAAPIRValue(IRB, 0),
          IRB.CreateAnd(
              FlagLoad,
              getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAlias))));

      if (RD.Racer.isValid()) {
        for (const Value *RObj : stableObjectOrder(RacerObjects)) {
          LLVM_DEBUG(dbgs() << "  Racer Object " << *RObj << "\n");
          // If the racer object matches Obj, there's no need to check a flag.
          if (RObj == Obj) {
            MAAPChk = IRB.getFalse();
            continue;
          }

          // Check if Loc and the racer object may alias.
          if (!AA->alias(Loc, MemoryLocation::getBeforeOrAfter(RObj)))
            continue;

          if (!LocalMAAPs.count(RObj)) {
            LLVM_DEBUG(dbgs() << "No local MAAP found for racer object " << RObj
                              << "\n");
            MAAPChk = IRB.getFalse();
            continue;
          }

          Value *FlagLoad = readMAAPVal(LocalMAAPs[RObj], IRB);
          LLVM_DEBUG(dbgs() << "  FlagLoad " << *FlagLoad << "\n");
          Value *LocalCheck;
          Value *FlagCheck = IRB.CreateAnd(
              FlagLoad, getMAAPIRValue(IRB, raceTypeToFlagVal(RD.Type)));
          LLVM_DEBUG(dbgs() << "  FlagCheck " << *FlagCheck << "\n");
          LocalCheck = IRB.CreateICmpEQ(getMAAPIRValue(IRB, 0), FlagCheck);
          LLVM_DEBUG(dbgs() << "  LocalCheck " << *LocalCheck << "\n");

          // Add the check.
          Value *RObjNoAliasFlag = IRB.CreateAnd(
              FlagLoad,
              getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAlias)));
          Value *RObjNoAliasCheck =
              IRB.CreateICmpNE(getMAAPIRValue(IRB, 0), RObjNoAliasFlag);
          MAAPChk = IRB.CreateAnd(
              MAAPChk,
              IRB.CreateOr(IRB.CreateOr(NoAliasCheck, RObjNoAliasCheck),
                           LocalCheck));
        }
      }

      // Check the function arguments that might alias this object.
      for (Argument &Arg : F->args()) {
        // Ignore non-pointer arguments
        if (!Arg.getType()->isPtrOrPtrVectorTy())
          continue;
        // Ignore any arguments that match checked objects.
        if (&Arg == Obj)
          continue;
        // Check if Loc and Arg may alias.
        if (!AA->alias(Loc, MemoryLocation::getBeforeOrAfter(&Arg)))
          continue;
        // If we have no local MAAP information about the argument, give up.
        if (!LocalMAAPs.count(&Arg)) {
          LLVM_DEBUG(dbgs() << "No local MAAP found for arg " << Arg << "\n");
          return IRB.getFalse();
        }

        LLVM_DEBUG(dbgs() << "  Argument " << Arg << "\n");

        // Incorporate the MAAP value for this argument if we don't have
        // a dynamic no-alias bit set.
        Value *FlagLoad = readMAAPVal(LocalMAAPs[&Arg], IRB);
        Value *FlagCheck;
        FlagCheck = IRB.CreateAnd(
            FlagLoad, getMAAPIRValue(IRB, raceTypeToFlagVal(RD.Type)));
        Value *LocalCheck = IRB.CreateICmpEQ(getMAAPIRValue(IRB, 0), FlagCheck);

        Value *ArgNoAliasFlag = IRB.CreateAnd(
            FlagLoad,
            getMAAPIRValue(IRB, static_cast<unsigned>(MAAPValue::NoAlias)));
        Value *ArgNoAliasCheck =
            IRB.CreateICmpNE(getMAAPIRValue(IRB, 0), ArgNoAliasFlag);
        MAAPChk = IRB.CreateAnd(
            MAAPChk, IRB.CreateOr(IRB.CreateOr(NoAliasCheck, ArgNoAliasCheck),
                                  LocalCheck));
      }
    }
  }
  return MAAPChk;
}

bool CilkSanitizerImpl::Instrumentor::performDelayedInstrumentation() {
  bool Result = false;
  // Handle delayed simple instructions
  for (Instruction *I : DelayedSimpleInsts) {
    assert((RI.mightRaceViaAncestor(I) || RI.mightRaceLocally(I)) &&
           "Delayed instrumentation is not local race or race via ancestor");
    IRBuilder<> IRB(I);
    DebugLoc Loc = searchForDebugLoc(I);

    if (MAAPChecks) {
      Value *MAAPChk = getMAAPCheck(I, IRB);
      if (MAAPChk != IRB.getFalse()) {
        Instruction *CheckTerm =
            SplitBlockAndInsertIfThen(IRB.CreateICmpEQ(MAAPChk, IRB.getFalse()),
                                      I, false, nullptr, &DTU, &LI);
        IRB.SetInsertPoint(CheckTerm);
      }
    }
    if (Loc)
      IRB.SetCurrentDebugLocation(Loc);
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      Result |= CilkSanImpl.instrumentLoadOrStore(I, IRB);
    else if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
      Result |= CilkSanImpl.instrumentAtomic(I, IRB);
    else
      dbgs() << "[Cilksan] Unknown simple instruction: " << *I << "\n";
  }

  // Handle delayed memory intrinsics
  for (auto &MemIntrinOp : DelayedMemIntrinsics) {
    Instruction *I = MemIntrinOp.first;
    assert((RI.mightRaceViaAncestor(I)  || RI.mightRaceLocally(I)) &&
           "Delayed instrumentation is not local race or race via ancestor");
    unsigned OperandNum = MemIntrinOp.second;
    IRBuilder<> IRB(I);
    DebugLoc Loc = searchForDebugLoc(I);

    if (MAAPChecks) {
      Value *MAAPChk = getMAAPCheck(I, IRB, OperandNum);
      if (MAAPChk != IRB.getFalse()) {
        Instruction *CheckTerm =
            SplitBlockAndInsertIfThen(IRB.CreateICmpEQ(MAAPChk, IRB.getFalse()),
                                      I, false, nullptr, &DTU, &LI);
        IRB.SetInsertPoint(CheckTerm);
      }
    }
    if (Loc)
      IRB.SetCurrentDebugLocation(Loc);
    Result |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, OperandNum, IRB);
  }
  return Result;
}

// Helper function to walk the hierarchy of tasks containing BasicBlock BB to
// get the top-level task in loop L that contains BB.
static Task *getTopLevelTaskFor(BasicBlock *BB, Loop *L, TaskInfo &TI) {
  Task *T = TI.getTaskFor(BB);
  // Return null if we don't find a task for BB contained in L.
  if (!T || !L->contains(T->getEntry()))
    return nullptr;

  // Walk up the tree of tasks until we discover a task containing BB that is
  // outside of L.
  while (L->contains(T->getParentTask()->getEntry()))
    T = T->getParentTask();

  return T;
}

void CilkSanitizerImpl::Instrumentor::getDetachesForCoalescedInstrumentation(
    SmallPtrSetImpl<Instruction *> &LoopInstToHoist,
    SmallPtrSetImpl<Instruction *> &LoopInstToSink) {
  // Determine detaches to instrument for the coalesced instrumentation.
  for (Instruction *I : inProgramOrder(LoopInstToHoist)) {
    Loop *L = LI.getLoopFor(I->getParent());
    // Record the detaches for the loop preheader, where the coalesced
    // instrumentation will be inserted.
    getDetachesForInstruction(L->getLoopPreheader()->getTerminator());
  }
  for (auto &NC : CilkSanImpl.NestChecks)
    getDetachesForInstruction(NC.insertPt());
  for (Instruction *I : inProgramOrder(LoopInstToSink)) {
    Loop *L = LI.getLoopFor(I->getParent());
    SmallVector<BasicBlock *, 4> ExitBlocks;
    L->getUniqueExitBlocks(ExitBlocks);
    for (BasicBlock *ExitBB : ExitBlocks) {
      if (getTopLevelTaskFor(ExitBB, L, TI))
        // Skip any exit blocks in a Tapir task inside the loop.  These exit
        // blocks lie on exception-handling paths, and to handle these blocks,
        // it suffices to insert instrumentation in the unwind destination of
        // the corresponding detach, which must also be a loop-exit block.
        continue;

      // Record the detaches for the exit block, where the coalesced
      // instrumentation will be inserted.
      getDetachesForInstruction(ExitBB->getTerminator());
    }
  }
}

bool CilkSanitizerImpl::Instrumentor::instrumentAncillaryInstructions(
    SmallPtrSetImpl<Instruction *> &Allocas,
    SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
    SmallPtrSetImpl<Instruction *> &FreeCalls,
    DenseMap<Value *, unsigned> &SyncRegNums,
    DenseMap<BasicBlock *, unsigned> &SRCounters, const DataLayout &DL) {
  bool Result = false;
  SmallSetVector<SyncInst *, 4> Syncs;
  SmallSetVector<Loop *, 4> Loops;
  SmallPtrSet<Instruction *, 16> InstrumentedAllocFns;

  // Instrument allocas and allocation-function calls that may be involved in a
  // race.
  for (Instruction *I : inProgramOrder(Allocas)) {
    if (CilkSanImpl.ObjectMRForRace.count(I) ||
        CilkSanImpl.lookupPointerMayBeCaptured(I)) {
      CilkSanImpl.instrumentAlloca(I, TI);
      getDetachesForInstruction(I);
      Result = true;
    }
  }
  for (Instruction *I : inProgramOrder(AllocationFnCalls)) {
    // Instrument any allocation-function calls that may allocate memory
    // involved in a race.
    //
    // Note that, without MAAP checks, we must be more conservative about
    // considering what memory allocations might be involved in checking for
    // races.  For example, suppose a function call in a loop uses memory that
    // is malloc'd and free'd within that loop.  Static analysis might determine
    // no race is possible on that memory, but a MAAP check is needed to
    // communicate that static information to the function at runtime in order
    // to avoid dynamic checks on the same location returned by repeated calls
    // to malloc.

    // FIXME: This test won't identify posix_memalign calls as needing
    // instrumentation, because posix_memalign modifies a pointer to the pointer
    // to the object.
    if (!MAAPChecks || CilkSanImpl.ObjectMRForRace.count(I) ||
        CilkSanImpl.lookupPointerMayBeCaptured(I)) {
      CilkSanImpl.instrumentAllocationFn(I, DT, TLI);
      InstrumentedAllocFns.insert(I);
      getDetachesForInstruction(I);
      Result = true;
    }
  }
  for (Instruction *I : inProgramOrder(FreeCalls)) {
    // The first argument of the free call is the pointer.
    Value *Ptr = I->getOperand(0);
    // If the pointer corresponds to an allocation function call in this
    // function, or if the pointer is involved in a race, then instrument it.
    if (Instruction *PtrI = dyn_cast<Instruction>(Ptr)) {
      if (InstrumentedAllocFns.count(PtrI)) {
        CilkSanImpl.instrumentFree(I, TLI);
        getDetachesForInstruction(I);
        Result = true;
        continue;
      }
    }
    if (RI.ObjectInvolvedInRace(Ptr) ||
        CilkSanImpl.unknownObjectUses(Ptr, &LI, TLI)) {
      CilkSanImpl.instrumentFree(I, TLI);
      getDetachesForInstruction(I);
      Result = true;
    }
  }

  // Instrument detaches
  for (DetachInst *DI : Detaches) {
    CilkSanImpl.instrumentDetach(DI, SyncRegNums[DI->getSyncRegion()],
                                 SRCounters[DI->getDetached()], DT, TI, LI);
    Result = true;
    // Get syncs associated with this detach
    for (SyncInst *SI : CilkSanImpl.DetachToSync[DI])
      Syncs.insert(SI);

    if (CilkSanImpl.Options.InstrumentLoops) {
      // Get any loop associated with this detach.
      Loop *L = LI.getLoopFor(DI->getParent());
      if (spawnsTapirLoopBody(DI, LI, TI))
        Loops.insert(L);
    }
  }

  // Instrument associated syncs
  for (SyncInst *SI : Syncs)
    CilkSanImpl.instrumentSync(SI, SyncRegNums[SI->getSyncRegion()]);

  if (CilkSanImpl.Options.InstrumentLoops) {
    // Recursively instrument all loops
    for (Loop *L : Loops)
      CilkSanImpl.instrumentTapirLoop(*L, TI, SyncRegNums);
  }

  return Result;
}

// Helper function to get a value for the runtime trip count of the given loop.
// True if L is not a Tapir loop and holds no detach, reattach or sync, so all
// of its iterations run in one strand: two of its accesses can only race
// through strands outside it.
static bool isSerialStrandLoop(const Loop *L, TaskInfo &TI) {
  if (getTaskIfTapirLoop(L, &TI))
    return false;
  for (const BasicBlock *BB : L->blocks()) {
    const Instruction *T = BB->getTerminator();
    if (isa<DetachInst>(T) || isa<ReattachInst>(T) || isa<SyncInst>(T))
      return false;
  }
  return true;
}

static const SCEV *getRuntimeTripCount(Loop &L, ScalarEvolution *SE,
                                       bool IsTapirLoop) {
  BasicBlock *Latch = L.getLoopLatch();

  // The exit count from the latch is sufficient for Tapir loops, because early
  // exits from such loops are handled in a special manner.  For other loops, we
  // use the backedge taken count.
  const SCEV *BECountSC =
      IsTapirLoop ? SE->getExitCount(&L, Latch) : SE->getBackedgeTakenCount(&L);
  if (isa<SCEVCouldNotCompute>(BECountSC) ||
      !BECountSC->getType()->isIntegerTy()) {
    LLVM_DEBUG(dbgs() << "Could not compute exit block SCEV\n");
    return SE->getCouldNotCompute();
  }

  // Add 1 since the backedge count doesn't include the first loop iteration.
  const SCEV *TripCountSC =
      SE->getAddExpr(BECountSC, SE->getConstant(BECountSC->getType(), 1));
  if (isa<SCEVCouldNotCompute>(TripCountSC)) {
    LLVM_DEBUG(dbgs() << "Could not compute trip count SCEV.\n");
    return SE->getCouldNotCompute();
  }

  return TripCountSC;
}

// Helper function to find where in the given basic block to insert coalesced
// instrumentation.
static Instruction *getLoopBlockInsertPt(BasicBlock *BB, FunctionCallee LoopHook,
                                         bool AfterHook) {
  // BasicBlock *PreheaderBB = L->getLoopPreheader();
  for (Instruction &I : *BB)
    if (CallBase *CB = dyn_cast<CallBase>(&I))
      if (const Function *Called = CB->getCalledFunction())
        if (Called == LoopHook.getCallee()) {
          // We found a call to the specified hook.  Pick an insertion point
          // with respect to it.
          if (AfterHook)
            return &*CB->getIterator()->getNextNode();
          return CB;
        }

  if (AfterHook)
    return &*BB->getFirstInsertionPt();
  return BB->getTerminator();
}

// TODO: Maybe to avoid confusion with CilkSanImpl.Options.InstrumentLoops
// (which is unrelated to this), rename this to involve the word "hoist" or something.
bool CilkSanitizerImpl::Instrumentor::instrumentLoops(
    SmallPtrSetImpl<Instruction *> &LoopInstToHoist,
    SmallPtrSetImpl<Instruction *> &LoopInstToSink,
    SmallPtrSetImpl<const Loop *> &TapirLoops, ScalarEvolution *SE) {
  bool Result = false;

  // First insert computation for the hook arguments for all instructions to
  // hoist or sink coalesced instrumentation.  We do this before inserting the
  // hook calls themselves, so that changes to the CFG -- specifically, from
  // inserting MAAP checks -- do not disrupt any function analyses we need.

  // Map instructions in the loop to address and range arguments for coalesced
  // instrumentation.
  DenseMap<Instruction *, std::pair<Value *, Value *>> HoistedHookArgs;
  // Compute arguments for coalesced instrumentation hoisted to before the loop.
  for (Instruction *I : inProgramOrder(LoopInstToHoist)) {
    // Get the insertion point in the preheader of the loop.
    Loop *L = LI.getLoopFor(I->getParent());
    assert(L->getLoopPreheader() && "No preheader for loop");
    Instruction *InsertPt =
        getLoopBlockInsertPt(L->getLoopPreheader(), CilkSanImpl.CsanBeforeLoop,
                             /*AfterHook*/ false);

    // TODO: Unify this SCEV computation with the similar computation for
    // instructions in LoopInstToSink.

    // Get the SCEV describing this instruction's pointer
    const SCEV *V = SE->getSCEV(getLoadStorePointerOperand(I));
    const SCEVAddRecExpr *SrcAR = dyn_cast<SCEVAddRecExpr>(V);

    // Get the stride
    const SCEV *StrideExpr = SrcAR->getStepRecurrence(*SE);
    assert(!isa<SCEVCouldNotCompute>(StrideExpr) &&
           "Stride should be computable");
    bool NegativeStride = SE->isKnownNegative(StrideExpr);
    if (NegativeStride)
      StrideExpr = SE->getNegativeSCEV(StrideExpr);

    // Get the first address accessed.
    const SCEV *FirstAddr = SrcAR->getStart();

    // Get the last address accessed.
    BasicBlock *Latch = L->getLoopLatch();
    const SCEV *BECount = TapirLoops.count(L) ? SE->getExitCount(L, Latch)
                                              : SE->getBackedgeTakenCount(L);
    const SCEV *LastAddr = SrcAR->evaluateAtIteration(BECount, *SE);

    // Get the size (number of bytes) of the address range accessed.
    const SCEV *RangeExpr = NegativeStride
                                ? SE->getMinusSCEV(FirstAddr, LastAddr)
                                : SE->getMinusSCEV(LastAddr, FirstAddr);
    RangeExpr = SE->getAddExpr(RangeExpr, StrideExpr);

    // Get the start (lowest address) of the address range accessed.
    const SCEV *Addr = NegativeStride ? LastAddr : FirstAddr;

    // Get instructions for calculating address range
    const DataLayout &DL = CilkSanImpl.M.getDataLayout();
    LLVMContext &Ctx = CilkSanImpl.M.getContext();
    SCEVExpander Expander(*SE, DL, "cilksan");

    Value *AddrVal =
        Expander.expandCodeFor(Addr, PointerType::getUnqual(Ctx), InsertPt);
    Value *RangeVal =
        Expander.expandCodeFor(RangeExpr, Type::getInt64Ty(Ctx), InsertPt);
    HoistedHookArgs[I] = std::make_pair(AddrVal, RangeVal);
  }

  // Addresses for the checks hoisted before loop nests, and for symbolic
  // nests, lengths and rows.
  SmallVector<Value *, 8> NestAddrs;
  SmallVector<Value *, 8> NestLens;
  SmallVector<SmallVector<std::pair<Value *, Value *>, 2>, 8> NestRows;
  for (auto &NC : CilkSanImpl.NestChecks) {
    if (NC.AddrV) {
      NestAddrs.push_back(NC.AddrV);
      NestLens.push_back(NC.LenV);
      NestRows.push_back(NC.RowsV);
      continue;
    }
    Instruction *InsertPt =
        NC.At ? NC.At
              : getLoopBlockInsertPt(NC.O->getLoopPreheader(),
                                     CilkSanImpl.CsanBeforeLoop,
                                     /*AfterHook*/ false);
    SCEVExpander Expander(*SE, CilkSanImpl.M.getDataLayout(), "cilksan");
    NestAddrs.push_back(Expander.expandCodeFor(
        NC.Addr, PointerType::getUnqual(CilkSanImpl.M.getContext()), InsertPt));
    Type *I64 = Type::getInt64Ty(CilkSanImpl.M.getContext());
    NestLens.push_back(
        NC.LenS ? Expander.expandCodeFor(NC.LenS, I64, InsertPt) : nullptr);
    SmallVector<std::pair<Value *, Value *>, 2> Rows;
    for (auto [Step, N] : NC.Rows)
      Rows.push_back({Expander.expandCodeFor(Step, I64, InsertPt),
                      Expander.expandCodeFor(N, I64, InsertPt)});
    NestRows.push_back(std::move(Rows));
  }

  // Map pairs of instruction and loop-exit to address and range arguments for
  // coalesced instrumentation.
  DenseMap<std::pair<Instruction *, BasicBlock *>, std::pair<Value *, Value *>>
      SunkHookArgs;
  // Map to track which loops we have already created counters for
  SmallMapVector<Loop*, Value*, 8> LoopToCounterMap;
  // Compute arguments for coalesced instrumentation sunk after the loop.
  for (Instruction *I : inProgramOrder(LoopInstToSink)) {
    // Get the loop
    Loop *L = LI.getLoopFor(I->getParent());

    // Add a counter to count the number of iterations executed in this loop.
    // In particular, this count will record the number of times the backedge of
    // the loop is taken.
    if (!LoopToCounterMap.count(L)) {
      assert(L->getLoopPreheader() && "No preheader for loop");
      assert(L->getLoopLatch() && "No unique latch for loop");
      IRBuilder<> IRB(&L->getHeader()->front());
      LLVMContext &Ctx = CilkSanImpl.M.getContext();

      PHINode *PN = IRB.CreatePHI(Type::getInt64Ty(Ctx), 2);
      PN->addIncoming(ConstantInt::getNullValue(Type::getInt64Ty(Ctx)),
                      L->getLoopPreheader());
      IRB.SetInsertPoint(&*L->getLoopLatch()->getFirstInsertionPt());
      Value *Add = IRB.CreateAdd(PN, ConstantInt::get(Type::getInt64Ty(Ctx), 1),
                                 "", true, true);
      PN->addIncoming(Add, L->getLoopLatch());
      LoopToCounterMap.insert(std::make_pair(L, PN));
    }

    // Get the counter for this loop.
    Value *Counter = LoopToCounterMap[L];

    // Get the SCEV describing this instruction's pointer
    const SCEV *V = SE->getSCEV(getLoadStorePointerOperand(I));
    const SCEVAddRecExpr *SrcAR = dyn_cast<SCEVAddRecExpr>(V);

    // Get the stride
    const SCEV *StrideExpr = SrcAR->getStepRecurrence(*SE);
    assert(!isa<SCEVCouldNotCompute>(StrideExpr) &&
           "Stride should be computable");
    bool NegativeStride = SE->isKnownNegative(StrideExpr);
    if (NegativeStride)
      StrideExpr = SE->getNegativeSCEV(StrideExpr);

    // Get the first address accessed.
    const SCEV *FirstAddr = SrcAR->getStart();

    // Get the last address accessed, based on the counter value..
    const SCEV *BECount = SE->getUnknown(Counter);
    const SCEV *LastAddr = SrcAR->evaluateAtIteration(BECount, *SE);

    // Get the size (number of bytes) of the address range accessed.
    const SCEV *RangeExpr = NegativeStride
                                ? SE->getMinusSCEV(FirstAddr, LastAddr)
                                : SE->getMinusSCEV(LastAddr, FirstAddr);
    RangeExpr = SE->getAddExpr(RangeExpr, StrideExpr);
    // Get the start (lowest address) of the address range accessed.
    const SCEV *Addr = NegativeStride ? LastAddr : FirstAddr;

    // Expand SCEV's into instructions for calculating the coalesced hook
    // arguments in each exit block.
    LLVMContext &Ctx = CilkSanImpl.M.getContext();
    const DataLayout &DL = CilkSanImpl.M.getDataLayout();
    SCEVExpander Expander(*SE, DL, "cilksan");
    SmallVector<BasicBlock *, 4> ExitBlocks;
    L->getUniqueExitBlocks(ExitBlocks);
    for (BasicBlock *ExitBB : ExitBlocks) {
      if (getTopLevelTaskFor(ExitBB, L, TI))
        // Skip any exit blocks in a Tapir task inside the loop.  These exit
        // blocks lie on exception-handling paths, and to handle these blocks,
        // it suffices to insert instrumentation in the unwind destination of
        // the corresponding detach, which must also be a loop-exit block.
        continue;

      // Instruction *InsertPt = &*ExitBB->getFirstInsertionPt();
      Instruction *InsertPt =
          getLoopBlockInsertPt(ExitBB, CilkSanImpl.CsanAfterLoop,
                               /*AfterHook*/ true);
      Value *AddrVal =
          Expander.expandCodeFor(Addr, PointerType::getUnqual(Ctx), InsertPt);
      Value *RangeVal =
          Expander.expandCodeFor(RangeExpr, Type::getInt64Ty(Ctx), InsertPt);

      assert(isa<Instruction>(RangeVal) &&
             "Expected computation of RangeVal to produce an instruction.");
      SunkHookArgs[std::make_pair(I, ExitBB)] =
          std::make_pair(AddrVal, RangeVal);
    }
  }

  // Now insert coalesced instrumentation, including relevant MAAP checks.
  //
  // TODO: For now, we only handle LoadInst and StoreInst.  Add other operations
  // later, such as atomics and memory intrinsics.

  // Insert coalesced instrumentation hoisted before the loop.
  for (Instruction *I : inProgramOrder(LoopInstToHoist)) {
    LLVM_DEBUG(dbgs() << "Loop instruction for hoisting instrumentation: " << *I
                      << "\n");

    // Get the local ID of this instruction.
    uint64_t LocalId;
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      uint64_t LoadId = CilkSanImpl.LoadFED.add(*LI);

      // TODO: Don't recalculate underlying objects
      uint64_t LoadObjId = CilkSanImpl.LoadObj.add(
          *LI,
          CilkSanImpl.lookupUnderlyingObject(getLoadStorePointerOperand(LI)));
      assert(LoadId == LoadObjId &&
             "Load received different ID's in FED and object tables.");
      LocalId = LoadId;
      // Update the statistic here, since we're guaranteed to insert the hook at
      // this point.
      ++NumHoistedInstrumentedReads;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      uint64_t StoreId = CilkSanImpl.StoreFED.add(*SI);

      // TODO: Don't recalculate underlying objects
      uint64_t StoreObjId = CilkSanImpl.StoreObj.add(
          *SI,
          CilkSanImpl.lookupUnderlyingObject(getLoadStorePointerOperand(SI)));
      assert(StoreId == StoreObjId &&
             "Store received different ID's in FED and object tables.");
      LocalId = StoreId;
      // Update the statistic here, since we're guaranteed to insert the hook at
      // this point.
      ++NumHoistedInstrumentedWrites;
    } else
      llvm_unreachable("Unexpected instruction to hoist instrumentation.");

    // For now, there shouldn't be a reason to return false since we already
    // verified the size, stride, and tripcount.
    Loop *L = LI.getLoopFor(I->getParent());
    Instruction *InsertPt =
        getLoopBlockInsertPt(L->getLoopPreheader(), CilkSanImpl.CsanBeforeLoop,
                             /*AfterLoop*/ false);
    IRBuilder<> IRB(InsertPt);
    if (MAAPChecks) {
      Value *MAAPChk = getMAAPCheck(I, IRB);
      if (MAAPChk != IRB.getFalse()) {
        Instruction *CheckTerm =
            SplitBlockAndInsertIfThen(IRB.CreateICmpEQ(MAAPChk, IRB.getFalse()),
                                      InsertPt, false, nullptr, &DTU, &LI);
        IRB.SetInsertPoint(CheckTerm);
      }
    }
    IRB.SetCurrentDebugLocation(searchForDebugLoc(I));
    CilkSanImpl.instrumentLoadOrStoreHoisted(
        I, HoistedHookArgs[I].first, HoistedHookArgs[I].second, IRB, LocalId);
    Result = true;
  }

  // Insert the checks hoisted before loop nests. A nest's runs of one kind
  // share their representative access, which is registered once.
  DenseMap<Instruction *, uint64_t> NestIds;
  for (size_t N = 0; N < CilkSanImpl.NestChecks.size(); ++N) {
    auto &NC = CilkSanImpl.NestChecks[N];
    Instruction *I = NC.Rep;
    uint64_t LocalId;
    if (auto It = NestIds.find(I); It != NestIds.end()) {
      LocalId = It->second;
      if (isa<LoadInst>(I))
        ++NumHoistedInstrumentedReads;
      else
        ++NumHoistedInstrumentedWrites;
    } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      LocalId = CilkSanImpl.LoadFED.add(*LI);
      uint64_t ObjId = CilkSanImpl.LoadObj.add(
          *LI,
          CilkSanImpl.lookupUnderlyingObject(getLoadStorePointerOperand(LI)));
      assert(LocalId == ObjId &&
             "Load received different ID's in FED and object tables.");
      (void)ObjId;
      ++NumHoistedInstrumentedReads;
    } else {
      StoreInst *SI = cast<StoreInst>(I);
      LocalId = CilkSanImpl.StoreFED.add(*SI);
      uint64_t ObjId = CilkSanImpl.StoreObj.add(
          *SI,
          CilkSanImpl.lookupUnderlyingObject(getLoadStorePointerOperand(SI)));
      assert(LocalId == ObjId &&
             "Store received different ID's in FED and object tables.");
      (void)ObjId;
      ++NumHoistedInstrumentedWrites;
    }
    NestIds[I] = LocalId;
    Instruction *InsertPt =
        NC.At ? NC.At
              : getLoopBlockInsertPt(NC.O->getLoopPreheader(),
                                     CilkSanImpl.CsanBeforeLoop,
                                     /*AfterLoop*/ false);
    IRBuilder<> IRB(InsertPt);
    if (!NC.Preds.empty()) {
      // The accesses run only when the nest's guarding conditions hold.
      Value *Guard = IRB.getTrue();
      for (auto [Cond, Dir] : NC.Preds)
        Guard = IRB.CreateAnd(Guard, Dir ? Cond : IRB.CreateNot(Cond));
      InsertPt = SplitBlockAndInsertIfThen(Guard, InsertPt, false, nullptr,
                                           &DTU, &LI);
      IRB.SetInsertPoint(InsertPt);
    }
    if (MAAPChecks) {
      // Skip the check only when every access it covers could skip its own.
      Value *RepChk = getMAAPCheck(I, IRB);
      Value *MAAPChk = IRB.getTrue();
      for (Instruction *M : NC.Members) {
        Value *Chk = M == I ? RepChk : getMAAPCheck(M, IRB);
        if (Chk == IRB.getFalse()) {
          MAAPChk = Chk;
          break;
        }
        MAAPChk = IRB.CreateAnd(MAAPChk, Chk);
      }
      if (RepChk != IRB.getFalse()) {
        if (MAAPChk == IRB.getFalse())
          ++NumMergedMAAPLost;
        if (auto *RI = dyn_cast<Instruction>(RepChk); RI && RI->use_empty())
          RecursivelyDeleteTriviallyDeadInstructions(RI);
      }
      if (MAAPChk != IRB.getFalse()) {
        ++NumMergedMAAPGuarded;
        Instruction *CheckTerm =
            SplitBlockAndInsertIfThen(IRB.CreateICmpEQ(MAAPChk, IRB.getFalse()),
                                      InsertPt, false, nullptr, &DTU, &LI);
        IRB.SetInsertPoint(CheckTerm);
      }
    }
    IRB.SetCurrentDebugLocation(searchForDebugLoc(I));
    Value *Addr = NestAddrs[N];
    Value *Len = NestLens[N] ? NestLens[N] : IRB.getInt64(NC.Len);
    // Rows of a symbolic nest: a loop per row dimension, each running at
    // least once (every loop in the nest does). The label fetch comes first,
    // since TaskInfo doesn't know the new blocks.
    CallInst *Label = nullptr;
    if (!NestRows[N].empty() &&
        (isa<LoadInst>(I) ? CilkSanImpl.LargeLoadTakesLabel
                          : CilkSanImpl.LargeStoreTakesLabel))
      Label = CilkSanImpl.getCurOSLabelCall(IRB);
    for (auto [Step, Count] : NestRows[N]) {
      BasicBlock *Pre = IRB.GetInsertBlock();
      Instruction *SplitPt = &*IRB.GetInsertPoint();
      BasicBlock *Exit = SplitBlock(Pre, SplitPt, &DTU, &LI, nullptr,
                                    "cilksan.rows.exit");
      LLVMContext &Ctx = Pre->getContext();
      BasicBlock *Body = BasicBlock::Create(Ctx, "cilksan.rows", Pre->getParent(),
                                            Exit);
      Pre->getTerminator()->setSuccessor(0, Body);
      IRBuilder<> BIRB(Body);
      BIRB.SetCurrentDebugLocation(IRB.getCurrentDebugLocation());
      PHINode *Idx = BIRB.CreatePHI(BIRB.getInt64Ty(), 2, "cilksan.row");
      Idx->addIncoming(BIRB.getInt64(0), Pre);
      Value *Next = BIRB.CreateAdd(Idx, BIRB.getInt64(1));
      BranchInst *Back = BIRB.CreateCondBr(BIRB.CreateICmpULT(Next, Count),
                                           Body, Exit);
      Idx->addIncoming(Next, Body);
      DTU.applyUpdates({{DominatorTree::Insert, Pre, Body},
                        {DominatorTree::Insert, Body, Body},
                        {DominatorTree::Insert, Body, Exit},
                        {DominatorTree::Delete, Pre, Exit}});
      Loop *RowLoop = LI.AllocateLoop();
      if (Loop *Parent = LI.getLoopFor(Pre))
        Parent->addChildLoop(RowLoop);
      else
        LI.addTopLevelLoop(RowLoop);
      RowLoop->addBasicBlockToLoop(Body, LI);
      IRB.SetInsertPoint(Back);
      Addr = IRB.CreateGEP(IRB.getInt8Ty(), Addr, IRB.CreateMul(Idx, Step));
    }
    // A run can start at a lower alignment than Rep's, so claim none: the
    // runtime's fast path trusts an access no larger than its alignment not
    // to cross an alignment boundary.
    CilkSanImpl.LabelOverride = Label;
    CilkSanImpl.instrumentLoadOrStoreHoisted(I, Addr, Len, IRB, LocalId,
                                             Align(1));
    CilkSanImpl.LabelOverride = nullptr;
    Result = true;
  }

  // Insert coalesced instrumentation sunk after the loop.
  for (Instruction *I : inProgramOrder(LoopInstToSink)) {
    LLVM_DEBUG(dbgs() << "Loop instruction for sinking instrumentation: " << *I
                      << "\n");
    Loop *L = LI.getLoopFor(I->getParent());

    // Get the local ID of this instruction.  We do this computation early to
    // avoid recomputing the local ID once per exit block.
    uint64_t LocalId;
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      uint64_t LoadId = CilkSanImpl.LoadFED.add(*LI);

      // TODO: Don't recalculate underlying objects
      uint64_t LoadObjId = CilkSanImpl.LoadObj.add(
          *LI,
          CilkSanImpl.lookupUnderlyingObject(getLoadStorePointerOperand(LI)));
      assert(LoadId == LoadObjId &&
             "Load received different ID's in FED and object tables.");
      LocalId = LoadId;
      // Update the statistic here, since we're guaranteed to insert the hooks
      // at this point, and to avoid overcounting the number of instructions on
      // loops with multiple exits.
      ++NumSunkInstrumentedReads;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      uint64_t StoreId = CilkSanImpl.StoreFED.add(*SI);

      // TODO: Don't recalculate underlying objects
      uint64_t StoreObjId = CilkSanImpl.StoreObj.add(
          *SI,
          CilkSanImpl.lookupUnderlyingObject(getLoadStorePointerOperand(SI)));
      assert(StoreId == StoreObjId &&
             "Store received different ID's in FED and object tables.");
      LocalId = StoreId;
      // Update the statistic here, since we're guaranteed to insert the hooks
      // at this point, and to avoid overcounting the number of instructions on
      // loops with multiple exits.
      ++NumSunkInstrumentedWrites;
    } else
      llvm_unreachable("Unexpected instruction to sink instrumentation.");

    SmallVector<BasicBlock *, 4> ExitBlocks;
    L->getUniqueExitBlocks(ExitBlocks);
    for (BasicBlock *ExitBB : ExitBlocks) {
      if (getTopLevelTaskFor(ExitBB, L, TI))
        // Skip any exit blocks in a Tapir task inside the loop.  These exit
        // blocks lie on exception-handling paths, and to handle these blocks,
        // it suffices to insert instrumentation in the unwind destination of
        // the corresponding detach, which must also be a loop-exit block.
        continue;

      // After the loop, perform the coalesced read/write.
      auto HookArgsKey = std::make_pair(I, ExitBB);

      // Insert the hook call after the computation of RangeVal.
      Instruction *InsertPt =
          cast<Instruction>(SunkHookArgs[HookArgsKey].second)
              ->getIterator()
              ->getNextNode();
      IRBuilder<> IRB(&*InsertPt);
      if (MAAPChecks) {
        Value *MAAPChk = getMAAPCheck(I, IRB);
        if (MAAPChk != IRB.getFalse()) {
          Instruction *CheckTerm = SplitBlockAndInsertIfThen(
              IRB.CreateICmpEQ(MAAPChk, IRB.getFalse()), &*InsertPt, false,
              nullptr, &DTU, &LI);
          IRB.SetInsertPoint(CheckTerm);
        }
      }
      IRB.SetCurrentDebugLocation(searchForDebugLoc(I));
      CilkSanImpl.instrumentLoadOrStoreHoisted(
          I, SunkHookArgs[HookArgsKey].first, SunkHookArgs[HookArgsKey].second,
          IRB, LocalId);
      Result = true;
    }
  }

  return Result;
}

bool CilkSanitizerImpl::instrumentLoadOrStoreHoisted(Instruction *I,
                                                     Value *Addr, Value *Size,
                                                     IRBuilder<> &IRB,
                                                     uint64_t LocalId,
                                                     MaybeAlign Alignment) {
  // The caller of this method is guaranteed to have computed the Addr and Size
  // values with the right type for the hook, so no additional type conversions
  // are needed.
  CsiLoadStoreProperty Prop;
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Prop.setAlignment(Alignment ? Alignment : MaybeAlign(LI->getAlign()));
    Prop.setIsThreadLocal(isThreadLocalObject(lookupUnderlyingObject(Addr)));
    // Instrument the load
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
    SmallVector<Value *, 5> Args = {CsiId, Addr, Size, Prop.getValue(IRB)};
    if (LargeLoadTakesLabel) {
      CallInst *CurLabCall = getCurOSLabelCall(IRB);
      Args.push_back(CurLabCall);
    }
    Instruction *Call = IRB.CreateCall(CsanLargeRead, Args);
    IRB.SetInstDebugLocation(Call);
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Prop.setAlignment(Alignment ? Alignment : MaybeAlign(SI->getAlign()));
    Prop.setIsThreadLocal(isThreadLocalObject(lookupUnderlyingObject(Addr)));
    // Instrument the store
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    SmallVector<Value *, 5> Args = {CsiId, Addr, Size, Prop.getValue(IRB)};
    if (LargeStoreTakesLabel) {
      CallInst *CurLabCall = getCurOSLabelCall(IRB);
      Args.push_back(CurLabCall);
    }
    Instruction *Call = IRB.CreateCall(CsanLargeWrite, Args);
    IRB.SetInstDebugLocation(Call);
  }
  return true;
}

static bool checkSanitizeCilkAttr(Function &F) {
  if (IgnoreSanitizeCilkAttr)
    return true;
  return F.hasFnAttribute(Attribute::SanitizeCilk);
}

bool CilkSanitizerImpl::setupFunction(Function &F, bool NeedToSetupCalls) {
  if (F.empty() || shouldNotInstrumentFunction(F) ||
      LinkedFromBitcode.count(&F) || !checkSanitizeCilkAttr(F)) {
    LLVM_DEBUG({
      dbgs() << "Skipping " << F.getName() << "\n";
      if (F.empty())
        dbgs() << "  Empty function\n";
      else if (shouldNotInstrumentFunction(F))
        dbgs() << "  Function should not be instrumented\n";
      else if (LinkedFromBitcode.count(&F))
        dbgs() << "  Function from linked-in bitcode\n";
      else if (!checkSanitizeCilkAttr(F))
        dbgs() << "  Function lacks sanitize_cilk attribute\n";
    });
    return false;
  }

  LLVM_DEBUG(dbgs() << "Setting up " << F.getName()
                    << " for instrumentation\n");

  // TODO: Move these steps into csi-setup pass.

  if (NeedToSetupCalls && Options.CallsMayThrow)
    // Promote calls to invokes to insert instrumentation in exception-handling
    // code.
    setupCalls(F);

  DominatorTree &DT = GetDomTree(F);
  LoopInfo &LI = GetLoopInfo(F);

  if (Options.InstrumentLoops)
    // Simplify loops to prepare for loop instrumentation
    for (Loop *L : LI)
      simplifyLoop(L, &DT, &LI, nullptr, nullptr, nullptr,
                   /* PreserveLCSSA */ false);

  // Canonicalize the CFG for instrumentation.
  setupBlocks(F, &DT, &LI);

  return true;
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
static void setInstrumentationDebugLoc(Function &Instrumented,
                                       Instruction *Call) {
  DISubprogram *Subprog = Instrumented.getSubprogram();
  if (Subprog) {
    LLVMContext &C = Instrumented.getParent()->getContext();
    Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
  }
}

CallInst *CilkSanitizerImpl::getCurOSLabelCall(IRBuilder<> &IRB) {
  if (LabelOverride)
    return LabelOverride;
  BasicBlock *BB = IRB.GetInsertBlock();
  const BasicBlock *Key = BB;
  BasicBlock *InsertBB = nullptr;

  if (CurTaskInfo && CurDomTree) {
    // Walk back to a block TaskInfo and the dominator tree both know about.
    // Blocks we created during instrumentation are in neither: the MAAP-guarded
    // "if (!maap)" blocks come from SplitBlockAndInsertIfThen, which is handed a
    // lazy DomTreeUpdater, so the tree has no node for them yet. Each such block
    // has exactly one predecessor -- the block it was split out of -- and the
    // split introduces a plain conditional branch, never a detach, reattach, or
    // sync, so the predecessor is in the same strand. Cap the walk so a
    // malformed chain cannot spin.
    BasicBlock *Known = BB;
    for (unsigned Steps = 0; Steps < 16; ++Steps) {
      if (CurTaskInfo->getSpindleFor(Known) && CurDomTree->getNode(Known))
        break;
      BasicBlock *Pred = Known->getUniquePredecessor();
      if (!Pred)
        break;
      Known = Pred;
    }
    Spindle *S = CurDomTree->getNode(Known) ? CurTaskInfo->getSpindleFor(Known)
                                            : nullptr;
    if (S) {
      // Only hoist when the spindle entry provably dominates this use. It
      // should by construction (a spindle has a single entry), but a bad hoist
      // here produces silently wrong labels, so we check rather than assume.
      // Check dominance against Known rather than BB: the tree may not have a
      // node for BB yet. Entry dominating Known implies it dominates BB, since
      // BB is reached from Known through the unique-predecessor chain above.
      BasicBlock *Entry = S->getEntry();
      // A spindle can run from a Tapir loop's latch (the continuation of its
      // detach) out through the loop exit, e.g. into a strip-mined loop's
      // serial epilog. Loop spawning outlines the loop later, so a label call
      // at an entry inside the loop must not serve a use outside it.
      auto LeavesTapirLoop = [&](BasicBlock *From, BasicBlock *To) {
        if (!CurLoopInfo)
          return false;
        for (Loop *L = CurLoopInfo->getLoopFor(From); L; L = L->getParentLoop())
          if (!L->contains(To) && getTaskIfTapirLoopStructure(L, CurTaskInfo))
            return true;
        return false;
      };
      if (Entry && CurDomTree->dominates(Entry, Known) &&
          !LeavesTapirLoop(Entry, Known)) {
        Key = Entry;
        InsertBB = Entry;
      }
    }
  }

  auto It = SpindleLabelCache.find(Key);
  if (It != SpindleLabelCache.end())
    return It->second;

  CallInst *Call;
  if (InsertBB) {
    BasicBlock::iterator IP = InsertBB->getFirstInsertionPt();
    IRBuilder<> EntryIRB(&*IP);
    // An inlinable call in a function with debug info must carry a !dbg
    // location or the verifier rejects the module. The call moves to the
    // spindle entry, so borrow a location from there; fall back to the
    // function scope when the entry block carries none (common at -O0).
    DebugLoc DL;
    for (Instruction &EntryI : *InsertBB)
      if (EntryI.getDebugLoc()) {
        DL = EntryI.getDebugLoc();
        break;
      }
    if (!DL)
      if (DISubprogram *SP = InsertBB->getParent()->getSubprogram())
        DL = DILocation::get(SP->getContext(), SP->getScopeLine(), 0, SP);
    EntryIRB.SetCurrentDebugLocation(DL);
    Call = EntryIRB.CreateCall(GetCurOSLabel);
  } else {
    // Not hoisted: fetch at the top of BB, so that the cached call dominates
    // every later use in BB. Uses are not instrumented in program order: K
    // emits a coalesced check after the last access it covers, before the
    // accesses in front of it (in the same block) are instrumented.
    IRBuilder<> TopIRB(BB, BB->getFirstInsertionPt());
    TopIRB.SetCurrentDebugLocation(IRB.getCurrentDebugLocation());
    Call = TopIRB.CreateCall(GetCurOSLabel);
  }
  Call->setCallingConv(CallingConv::PreserveMost);
  SpindleLabelCache[Key] = Call;
  return Call;
}

// Loop nests checked as a whole (see HoistNests). Nest O qualifies if it runs
// in one strand -- it is serial, holds no detach, reattach or sync, and calls
// nothing that might race -- and every loop in it has one exit, a constant
// trip count and, if it is not O, runs on every iteration of its parent. An
// access in O is covered if it runs on every iteration of its loop and its
// address is affine in the indices of its loops up to O, with constant steps,
// from a base fixed on entry to O. The bytes such accesses touch, from bases
// that differ by constants, are then known exactly: they are enumerated, and
// one check before O covers each run of written bytes, and of bytes read but
// not written (the strand's write covers its read). The order of checks
// within the strand doesn't change what they find. Accesses that don't
// qualify are checked as before.
void CilkSanitizerImpl::planNestChecks(LoopInfo &LI, ScalarEvolution &SE,
                                       DominatorTree &DT, TaskInfo &TI,
                                       RaceInfo &RI,
                                       SmallPtrSetImpl<Instruction *> &Covered) {
  // Outermost first: a qualifying nest's subloops are not tried on their own.
  SmallVector<Loop *, 8> Work(LI.begin(), LI.end());
  while (!Work.empty()) {
    Loop *O = Work.pop_back_val();
    if (!tryNest(O, LI, SE, DT, TI, RI, Covered))
      Work.append(O->begin(), O->end());
  }
}

// Why the checks of nest O can't move before it, as for a nest that runs in
// one strand, or null if they can: O needs a preheader, must run in one strand
// (isSerialStrandLoop), and must call nothing that might race.
const char *CilkSanitizerImpl::whyNotOneStrand(const Loop *O, TaskInfo &TI) {
  if (!O->getLoopPreheader() || !isSerialStrandLoop(O, TI))
    return "no preheader or not one strand";
  for (const BasicBlock *BB : O->blocks())
    for (const Instruction &I : *BB)
      // Intrinsics that touch no memory (fmuladd, umin, ...) are arithmetic.
      if (isa<CallBase>(I) && !isa<DbgInfoIntrinsic>(I) &&
          !(isa<IntrinsicInst>(I) && !I.mayReadOrWriteMemory()) &&
          !simpleCallCannotRace(I)) {
        LLVM_DEBUG(dbgs() << "  call: " << I << "\n");
        return "calls something that might race";
      }
  return nullptr;
}

// Checks the exact footprint of accesses As, whose offsets are from Base,
// for a nest (O, checks before it) or a straight-line stretch (O null, checks
// at At): one check per run of bytes written, and of bytes read but not
// written. A straight-line access is the zero-dimension case. The check of a
// run reports the first (FirstRep) or last access of its kind, and its
// members are the accesses with a byte in it, in order. Returns the number of
// checks, or 0, adding none, if the footprint spans more than MaxSpan bytes,
// marks more than MaxMarks instances or has more than MaxRuns runs.
unsigned CilkSanitizerImpl::coverClass(ArrayRef<FootprintAccess> As,
                                       ScalarEvolution &SE, const SCEV *Base,
                                       Loop *O, Instruction *At, bool FirstRep,
                                       int64_t MaxSpan, uint64_t MaxMarks,
                                       unsigned MaxRuns) {
  int64_t Lo = INT64_MAX, Hi = INT64_MIN;
  uint64_t Marks = 0;
  for (const FootprintAccess &A : As) {
    int64_t ALo = A.Off, AHi = A.Off + A.Size;
    uint64_t N = 1;
    for (auto [Step, T] : A.Steps) {
      int64_t Ext = Step * int64_t(T - 1);
      (Ext < 0 ? ALo : AHi) += Ext;
      N *= T;
    }
    Lo = std::min(Lo, ALo);
    Hi = std::max(Hi, AHi);
    Marks += N;
  }
  if (Hi - Lo > MaxSpan || Marks > MaxMarks) {
    LLVM_DEBUG(dbgs() << "  class too large: span " << (Hi - Lo) << "\n");
    return 0;
  }
  // Calls Fn with the offset from Lo of each instance of A.
  auto ForEachInstance = [&](const FootprintAccess &A,
                             function_ref<void(int64_t)> Fn) {
    SmallVector<unsigned, 4> Idx(A.Steps.size(), 0);
    for (;;) {
      int64_t Off = A.Off - Lo;
      for (size_t k = 0; k < Idx.size(); ++k)
        Off += A.Steps[k].first * int64_t(Idx[k]);
      Fn(Off);
      size_t k = 0;
      for (; k < Idx.size(); ++k) {
        if (++Idx[k] < A.Steps[k].second)
          break;
        Idx[k] = 0;
      }
      if (k == Idx.size())
        break;
    }
  };
  // Bytes read (1) and written (2), from Lo.
  std::vector<uint8_t> Map(Hi - Lo, 0);
  for (const FootprintAccess &A : As) {
    uint8_t Bit = isa<StoreInst>(A.I) ? 2 : 1;
    ForEachInstance(A, [&](int64_t Off) {
      for (int64_t b = 0; b < A.Size; ++b)
        Map[Off + b] |= Bit;
    });
  }
  SmallVector<ByteRun, 8> Runs = byteMapRuns(Map);
  if (Runs.size() > MaxRuns) {
    LLVM_DEBUG(dbgs() << "  class has " << Runs.size() << " runs\n");
    return 0;
  }
  Instruction *RepR = nullptr, *RepW = nullptr;
  for (const FootprintAccess &A : As) {
    Instruction *&Rep = isa<StoreInst>(A.I) ? RepW : RepR;
    if (!FirstRep || !Rep)
      Rep = A.I;
  }
  // The run of each byte, and the accesses with a byte in each run.
  std::vector<int32_t> RunOf(Hi - Lo, -1);
  for (size_t r = 0; r < Runs.size(); ++r)
    for (int64_t b = 0; b < std::get<1>(Runs[r]); ++b)
      RunOf[std::get<0>(Runs[r]) + b] = r;
  SmallVector<SmallSetVector<Instruction *, 4>, 8> RunMembers(Runs.size());
  for (const FootprintAccess &A : As)
    ForEachInstance(A, [&](int64_t Off) {
      for (int64_t b = 0; b < A.Size; ++b)
        RunMembers[RunOf[Off + b]].insert(A.I);
    });
  for (size_t r = 0; r < Runs.size(); ++r) {
    auto [B, N, W] = Runs[r];
    const SCEV *Addr = SE.getAddExpr(
        Base, SE.getConstant(SE.getEffectiveSCEVType(Base->getType()), Lo + B,
                             /*isSigned=*/true));
    NestChecks.push_back({W ? RepW : RepR, O, Addr, uint64_t(N), At});
    NestChecks.back().Members.append(RunMembers[r].begin(),
                                     RunMembers[r].end());
  }
  return Runs.size();
}

bool CilkSanitizerImpl::tryNest(Loop *O, LoopInfo &LI, ScalarEvolution &SE,
                                DominatorTree &DT, TaskInfo &TI, RaceInfo &RI,
                                SmallPtrSetImpl<Instruction *> &Covered) {
  auto Reject = [&](const char *Why) {
    LLVM_DEBUG(dbgs() << "Loop nest " << O->getHeader()->getName() << " in "
                      << O->getHeader()->getParent()->getName() << ": " << Why
                      << "\n");
    return false;
  };
  if (const char *Why = whyNotOneStrand(O, TI))
    return Reject(Why);
  // Every loop: one exit, a constant trip count, and for an inner loop, a
  // preheader that runs on every iteration of its parent.
  SmallVector<Loop *, 8> Loops = O->getLoopsInPreorder();
  DenseMap<const Loop *, unsigned> Trips;
  for (Loop *L : Loops) {
    unsigned T = SE.getSmallConstantTripCount(L);
    if (!T || !L->getExitingBlock() || !L->getLoopPreheader())
      return Reject("a loop without a constant trip count or one exit");
    Trips[L] = T;
    if (L != O) {
      ICFLoopSafetyInfo PSI;
      PSI.computeLoopSafetyInfo(L->getParentLoop());
      if (!PSI.isGuaranteedToExecute(*L->getLoopPreheader()->getTerminator(),
                                     &DT, &TI, L->getParentLoop()))
        return Reject("an inner loop that doesn't run on every iteration");
    }
  }

  // The accesses to cover, by base (up to a constant).
  struct Class {
    const SCEV *Base;
    SmallVector<FootprintAccess, 8> Accesses;
  };
  SmallVector<Class, 4> Classes;
  DenseMap<const Loop *, std::unique_ptr<ICFLoopSafetyInfo>> SafetyInfos;
  for (Loop *L : Loops) {
    auto SI = std::make_unique<ICFLoopSafetyInfo>();
    SI->computeLoopSafetyInfo(L);
    SafetyInfos[L] = std::move(SI);
  }
  for (BasicBlock *BB : O->blocks()) {
    Loop *Inner = LI.getLoopFor(BB);
    for (Instruction &I : *BB) {
      std::optional<int64_t> Size = hoistableAccessSize(I, RI, SE);
      if (!Size ||
          !SafetyInfos[Inner]->isGuaranteedToExecute(I, &DT, &TI, Inner))
        continue;
      // Peel one affine level per loop from Inner up to O.
      const SCEV *V = SE.getSCEV(getLoadStorePointerOperand(&I));
      FootprintAccess A{&I, 0, *Size, {}};
      bool OK = true;
      for (Loop *L = Inner;; L = L->getParentLoop()) {
        if (const auto *AR = dyn_cast<SCEVAddRecExpr>(V);
            AR && AR->getLoop() == L) {
          const auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
          if (!AR->isAffine() || !Step ||
              Step->getAPInt().getSignificantBits() > 32) {
            OK = false;
            break;
          }
          A.Steps.push_back({Step->getAPInt().getSExtValue(), Trips[L]});
          V = AR->getStart();
        } else if (!SE.isLoopInvariant(V, L)) {
          OK = false;
          break;
        }
        if (L == O)
          break;
      }
      if (!OK || !SE.isLoopInvariant(V, O) || !SE.isAvailableAtLoopEntry(V, O)) {
        LLVM_DEBUG(dbgs() << "  not affine with a fixed base: " << I << "\n");
        continue;
      }
      Class *C = nullptr;
      for (Class &K : Classes)
        if (std::optional<int64_t> Off = constOffset(SE, V, K.Base)) {
          A.Off = *Off;
          C = &K;
          break;
        }
      if (!C) {
        Classes.push_back({V, {}});
        C = &Classes.back();
      }
      C->Accesses.push_back(std::move(A));
    }
  }

  // Each class's exact footprint, as runs.
  bool Any = false;
  for (Class &K : Classes) {
    unsigned Checks = coverClass(K.Accesses, SE, K.Base, O, /*At=*/nullptr,
                                 /*FirstRep=*/false, int64_t(1) << 20,
                                 uint64_t(1) << 22, /*MaxRuns=*/32);
    if (!Checks)
      continue;
    for (FootprintAccess &A : K.Accesses)
      Covered.insert(A.I);
    NumNestChecks += Checks;
    NumNestAccesses += K.Accesses.size();
    Any = true;
    LLVM_DEBUG(dbgs() << "Hoisting " << Checks << " checks for "
                      << K.Accesses.size() << " accesses before loop nest "
                      << O->getHeader()->getName() << "\n");
  }
  return Any;
}

// Loop nests with symbolic bounds (see HoistSymbolicNests). As for tryNest,
// nest O runs in one strand and every loop in it has one exit, but a trip
// count need only be fixed on entry to O: each loop's backedge-taken count is
// invariant in O. An inner loop's preheader, and every covered access, must
// dominate the exiting block of its loop, so it runs exactly once per
// iteration, and every loop runs at least once whenever O is entered. An
// access is covered if its address is affine in the indices of its loops up
// to O, with steps and base fixed on entry to O. Its footprint is then a
// set of rows: dimensions whose step is at most the extent built so far
// (provably, as a constant, or by being equal to it) widen one contiguous
// range, and up to two others become loops of checks. Every checked byte is
// one the access touches, so the checks are exact, and as before their order
// within the strand doesn't change what they find. An access whose
// footprint doesn't shrink this way is checked as before.
//
// Finds the footprints of the accesses SizeOf gives a size for (in Prints),
// and for each access, how many versions of its loop it has (NumVersions;
// see LoopState). Returns false, finding none, if O doesn't qualify.
bool CilkSanitizerImpl::symbolicFootprints(
    Loop *O, LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT,
    TaskInfo &TI, function_ref<std::optional<int64_t>(Instruction &)> SizeOf,
    SmallVectorImpl<SymFootprint> &Prints,
    DenseMap<Instruction *, unsigned> &NumVersions) {
  auto Reject = [&](const char *Why) {
    LLVM_DEBUG(dbgs() << "Symbolic nest " << O->getHeader()->getName()
                      << " in " << O->getHeader()->getParent()->getName()
                      << ": " << Why << "\n");
    return false;
  };
  if (const char *Why = whyNotOneStrand(O, TI))
    return Reject(Why);
  Type *I64 = Type::getInt64Ty(O->getHeader()->getContext());
  // A value fixed on entry to O: defined before it.
  Instruction *Entry = O->getLoopPreheader()->getTerminator();
  auto FixedOnEntry = [&](Value *V) {
    auto *I = dyn_cast<Instruction>(V);
    return !I || (!O->contains(I) && DT.dominates(I, Entry));
  };
  // A condition computed in O from values fixed on entry, by instructions
  // without side effects (a select of two checks, say), is itself fixed on
  // entry: clone its computation before O. Returns null if it isn't.
  DenseMap<Value *, Value *> Hoisted;
  std::function<Value *(Value *, unsigned)> HoistCond = [&](Value *V,
                                                            unsigned Depth) {
    if (FixedOnEntry(V))
      return V;
    if (auto It = Hoisted.find(V); It != Hoisted.end())
      return It->second;
    auto *I = dyn_cast<Instruction>(V);
    if (!I || Depth > 4 || isa<PHINode>(I) || I->mayReadOrWriteMemory() ||
        !isSafeToSpeculativelyExecute(I))
      return (Value *)nullptr;
    SmallVector<Value *, 4> Ops;
    for (Value *Op : I->operands()) {
      Value *H = HoistCond(Op, Depth + 1);
      if (!H)
        return (Value *)nullptr;
      Ops.push_back(H);
    }
    Instruction *C = I->clone();
    for (unsigned k = 0; k < Ops.size(); ++k)
      C->setOperand(k, Ops[k]);
    C->insertBefore(Entry->getIterator());
    C->setName(I->getName() + ".cilksan");
    Hoisted[V] = C;
    return (Value *)C;
  };
  // Expanding a SCEV before O mustn't divide by a value that may be zero.
  auto SafeToExpand = [&](const SCEV *S) {
    return !SCEVExprContains(S, [](const SCEV *X) {
      const auto *D = dyn_cast<SCEVUDivExpr>(X);
      return D && !isa<SCEVConstant>(D->getRHS());
    });
  };
  SmallVector<Loop *, 8> Loops = O->getLoopsInPreorder();
  DenseMap<const Loop *, std::unique_ptr<ICFLoopSafetyInfo>> SafetyInfos;
  for (Loop *L : Loops) {
    auto SI = std::make_unique<ICFLoopSafetyInfo>();
    SI->computeLoopSafetyInfo(L);
    SafetyInfos[L] = std::move(SI);
  }
  using PredList = SmallVector<std::pair<Value *, bool>, 2>;
  // Conditions as an OR of exclusive clauses, each an AND of branch
  // conditions fixed on entry to O.
  using DNF = SmallVector<PredList, 2>;
  constexpr size_t kMaxClauses = 8;
  // A clause and'ed with another, or none if they contradict each other.
  auto Conj = [](const PredList &A, const PredList &B) -> std::optional<PredList> {
    PredList R = A;
    for (auto Lit : B) {
      auto It = find_if(R, [&](auto &X) { return X.first == Lit.first; });
      if (It == R.end())
        R.push_back(Lit);
      else if (It->second != Lit.second)
        return std::nullopt;
    }
    return R;
  };
  // Merges clauses that differ only in one condition's sense (A and c, A and
  // !c) into one (A), until none do: the exits of an unrolled vector loop,
  // one per possible trip count, together leave whenever it was entered.
  auto Simplify = [](DNF &R) {
    for (bool Changed = true; Changed;) {
      Changed = false;
      for (size_t a = 0; a < R.size() && !Changed; ++a)
        for (size_t b = a + 1; b < R.size() && !Changed; ++b) {
          if (R[a].size() != R[b].size())
            continue;
          int Diff = -1;
          bool OK = true;
          for (size_t k = 0; k < R[a].size() && OK; ++k) {
            auto It = find_if(R[b], [&](auto &X) {
              return X.first == R[a][k].first;
            });
            if (It == R[b].end())
              OK = false;
            else if (It->second != R[a][k].second) {
              if (Diff >= 0)
                OK = false;
              Diff = k;
            }
          }
          if (!OK || Diff < 0)
            continue;
          R[a].erase(R[a].begin() + Diff);
          R.erase(R.begin() + b);
          Changed = true;
        }
    }
  };
  // The conditions under which BB, in loop L, runs on an iteration of L (at
  // most once, as a block outside L's subloops): none if it always does.
  // Otherwise, exactly when one of the edges into it is taken; an edge from a
  // block of L is taken when that block runs and its branch goes this way,
  // and the edge out of a subloop with one exit when the subloop's preheader
  // runs (its trip count is known, so it ends). (The vectorizer leaves such
  // blocks when it unrolls a vector loop with a small symbolic trip count,
  // and around its epilogues.)
  DenseMap<BasicBlock *, std::optional<DNF>> BlockMemo;
  std::function<std::optional<DNF>(BasicBlock *, Loop *, unsigned)> BlockPreds;
  std::function<std::optional<DNF>(BasicBlock *, BasicBlock *, Loop *,
                                   unsigned)>
      EdgePreds;
  BlockPreds = [&](BasicBlock *BB, Loop *L,
                   unsigned Depth) -> std::optional<DNF> {
    if (auto It = BlockMemo.find(BB); It != BlockMemo.end())
      return It->second;
    std::optional<DNF> R;
    if (SafetyInfos[L]->isGuaranteedToExecute(*BB->getTerminator(), &DT, &TI,
                                              L) &&
        DT.dominates(BB, L->getExitingBlock())) {
      R = DNF{PredList{}};
    } else if (Depth < 32 && BB != L->getHeader()) {
      R = DNF{};
      SmallPtrSet<BasicBlock *, 4> Seen;
      for (BasicBlock *Q : predecessors(BB)) {
        if (!Seen.insert(Q).second)
          continue;
        auto E = EdgePreds(Q, BB, L, Depth + 1);
        if (!E || R->size() + E->size() > 8 * kMaxClauses) {
          R = std::nullopt;
          break;
        }
        R->append(E->begin(), E->end());
      }
      if (R) {
        Simplify(*R);
        if (R->size() > kMaxClauses)
          R = std::nullopt;
      }
    }
    BlockMemo[BB] = R;
    return R;
  };
  EdgePreds = [&](BasicBlock *Q, BasicBlock *BB, Loop *L,
                  unsigned Depth) -> std::optional<DNF> {
    if (!L->contains(Q))
      return std::nullopt;
    if (Loop *Sub = LI.getLoopFor(Q); Sub != L) {
      while (Sub->getParentLoop() != L)
        Sub = Sub->getParentLoop();
      if (Sub->getExitingBlock() != Q || Sub->getUniqueExitBlock() != BB ||
          !Sub->getLoopPreheader() ||
          !SE.hasLoopInvariantBackedgeTakenCount(Sub))
        return std::nullopt;
      return BlockPreds(Sub->getLoopPreheader(), L, Depth);
    }
    auto *Br = dyn_cast<BranchInst>(Q->getTerminator());
    if (!Br)
      return std::nullopt;
    auto P = BlockPreds(Q, L, Depth);
    if (!P || !Br->isConditional() || Br->getSuccessor(0) == Br->getSuccessor(1))
      return P;
    Value *Cond = HoistCond(Br->getCondition(), 0);
    if (!Cond)
      return std::nullopt;
    DNF R;
    for (PredList &C : *P)
      if (auto D = Conj(C, PredList{{Cond, Br->getSuccessor(0) == BB}}))
        R.push_back(std::move(*D));
    return R;
  };
  // Each loop that runs once on every iteration of its parent, at least
  // when some branch conditions fixed on entry to O go one way, has one
  // version per way in: the conditions under which it is entered that way
  // (exactly those, so they hold on every iteration of the parent or on
  // none), the values its preheader's phis then take (Map), and the trip
  // counts of it and its ancestors up to O. The vectorizer leaves loops like
  // this: a scalar loop entered from its runtime checks with a start of 0,
  // or after the vector loop, as its remainder.
  struct LoopState {
    PredList Preds;
    ValueToSCEVMapTy Map;
    DenseMap<const Loop *, const SCEV *> Trips;
    // Whether some way into this loop or an ancestor has no version, so the
    // loop may also run when no version's conditions hold.
    bool Partly = false;
  };
  constexpr size_t kMaxVersions = 8;
  DenseMap<const Loop *, SmallVector<LoopState, 2>> States;
  for (Loop *L : Loops) {
    BasicBlock *PH = L->getLoopPreheader();
    if (!L->getExitingBlock() || !PH) {
      if (L == O)
        return Reject("a loop without one exit");
      continue;
    }
    SmallVector<LoopState, 2> Vs;
    if (L == O) {
      Vs.emplace_back();
    } else {
      Loop *P = L->getParentLoop();
      auto PIt = States.find(P);
      if (PIt == States.end())
        continue;
      for (LoopState &PV : PIt->second) {
        // A preheader with one way in (or the parent's header) has no phis
        // to resolve: its own conditions give the versions.
        if (PH == P->getHeader() || PH->getUniquePredecessor()) {
          auto Bp = BlockPreds(PH, P, 0);
          if (!Bp)
            continue;
          for (PredList &D : *Bp)
            if (auto C = Conj(PV.Preds, D)) {
              LoopState V = PV;
              V.Preds = std::move(*C);
              Vs.push_back(std::move(V));
            }
          continue;
        }
        bool All = true;
        size_t First = Vs.size();
        SmallPtrSet<BasicBlock *, 4> Seen;
        for (BasicBlock *Pd : predecessors(PH)) {
          if (!Seen.insert(Pd).second)
            continue;
          auto E = EdgePreds(Pd, PH, P, 0);
          if (!E) {
            All = false;
            continue;
          }
          for (PredList &D : *E) {
            auto C = Conj(PV.Preds, D);
            if (!C)
              continue;
            LoopState V = PV;
            V.Preds = std::move(*C);
            for (PHINode &Phi : PH->phis())
              if (SE.isSCEVable(Phi.getType()))
                V.Map[&Phi] = SCEVParameterRewriter::rewrite(
                    SE.getSCEV(Phi.getIncomingValueForBlock(Pd)), SE, PV.Map);
            Vs.push_back(std::move(V));
          }
        }
        if (!All)
          for (size_t k = First; k < Vs.size(); ++k)
            Vs[k].Partly = true;
      }
    }
    // Each version's trip count, fixed on entry to O.
    bool Dropped = Vs.size() > kMaxVersions;
    if (Dropped)
      Vs.resize(kMaxVersions);
    SmallVector<LoopState, 2> Kept;
    for (LoopState &V : Vs) {
      const SCEV *BTC = SE.getBackedgeTakenCount(L);
      if (!isa<SCEVCouldNotCompute>(BTC))
        BTC = SCEVParameterRewriter::rewrite(BTC, SE, V.Map);
      if (isa<SCEVCouldNotCompute>(BTC) || !SE.isLoopInvariant(BTC, O) ||
          !SE.isAvailableAtLoopEntry(BTC, O) || !SafeToExpand(BTC) ||
          SE.getTypeSizeInBits(BTC->getType()) > 64) {
        if (L == O)
          return Reject("a trip count not fixed on entry");
        Dropped = true;
        continue;
      }
      V.Trips[L] =
          SE.getAddExpr(SE.getNoopOrZeroExtend(BTC, I64), SE.getOne(I64));
      Kept.push_back(std::move(V));
    }
    if (Dropped)
      for (LoopState &V : Kept)
        V.Partly = true;
    LLVM_DEBUG({
      dbgs() << "  loop " << L->getHeader()->getName() << ": " << Kept.size()
             << " versions" << (Dropped ? ", some dropped" : "") << "\n";
      for (LoopState &V : Kept) {
        dbgs() << "    " << (V.Partly ? "partly" : "fully") << " when";
        for (auto [C, D] : V.Preds)
          dbgs() << " " << (D ? "" : "!") << C->getName();
        dbgs() << ", trip " << *V.Trips[L] << "\n";
      }
    });
    if (!Kept.empty())
      States[L] = std::move(Kept);
  }
  if (Loops.size() == 1 && isa<SCEVConstant>(States[O][0].Trips[O]))
    return Reject("one loop with a constant trip count");

  // An access before grouping: Base + [0, Size) + sum of Step_k * [0, N_k),
  // in Depth loops.
  struct Proto {
    Instruction *I;
    const SCEV *Base;
    int64_t Size;
    SmallVector<std::pair<const SCEV *, const SCEV *>, 4> Dims;
    PredList Preds;
    unsigned Depth;
    bool Partly;
  };
  SmallVector<Proto, 16> Protos;
  for (BasicBlock *BB : O->blocks()) {
    Loop *Inner = LI.getLoopFor(BB);
    auto StIt = States.find(Inner);
    if (StIt == States.end())
      continue;
    for (Instruction &I : *BB) {
      std::optional<int64_t> Size = SizeOf(I);
      if (!Size)
        continue;
      // I runs once on every iteration of Inner, possibly under conditions
      // fixed on entry: its block does, and nothing before it in the block
      // can stop it.
      if (!all_of(make_range(BB->begin(), I.getIterator()),
                  [](Instruction &J) {
                    return isGuaranteedToTransferExecutionToSuccessor(&J);
                  }))
        continue;
      auto BPs = BlockPreds(BB, Inner, 0);
      if (!BPs)
        continue;
      // Each version of Inner, under each clause of BB's conditions.
      SmallVector<std::pair<LoopState *, PredList>, 4> Cases;
      for (LoopState &St : StIt->second)
        for (PredList &D : *BPs)
          if (auto C = Conj(St.Preds, D))
            Cases.push_back({&St, std::move(*C)});
      NumVersions[&I] = Cases.size();
      for (auto &[StP, CasePreds] : Cases) {
      LoopState &St = *StP;
      // Peel one affine level per loop from Inner up to O.
      const SCEV *V = SCEVParameterRewriter::rewrite(
          SE.getSCEV(getLoadStorePointerOperand(&I)), SE, St.Map);
      SmallVector<std::pair<const SCEV *, const SCEV *>, 4> Dims;
      bool OK = true;
      for (Loop *L = Inner;; L = L->getParentLoop()) {
        if (const auto *AR = dyn_cast<SCEVAddRecExpr>(V);
            AR && AR->getLoop() == L) {
          const SCEV *Step = AR->getStepRecurrence(SE);
          if (!AR->isAffine() || !SE.isLoopInvariant(Step, O) ||
              !SE.isAvailableAtLoopEntry(Step, O) || !SafeToExpand(Step) ||
              SE.getTypeSizeInBits(Step->getType()) > 64) {
            OK = false;
            break;
          }
          Step = SE.getNoopOrSignExtend(Step, I64);
          if (!Step->isZero())
            Dims.push_back({Step, St.Trips.lookup(L)});
          V = AR->getStart();
        } else if (!SE.isLoopInvariant(V, L)) {
          OK = false;
          break;
        }
        if (L == O)
          break;
      }
      if (!OK || !SE.isLoopInvariant(V, O) ||
          !SE.isAvailableAtLoopEntry(V, O) || !SafeToExpand(V)) {
        LLVM_DEBUG(dbgs() << "  not affine with a fixed base: " << I << "\n");
        break;
      }
      // Negative constant steps walk down from the last element.
      for (auto &[Step, N] : Dims)
        if (const auto *C = dyn_cast<SCEVConstant>(Step);
            C && C->getAPInt().isNegative()) {
          V = SE.getAddExpr(
              V, SE.getMulExpr(Step, SE.getMinusSCEV(N, SE.getOne(I64))));
          Step = SE.getNegativeSCEV(Step);
        }
      Protos.push_back({&I, V, *Size, {Dims.begin(), Dims.end()}, CasePreds,
                        LI.getLoopDepth(BB) - O->getLoopDepth() + 1,
                        St.Partly});
      }
    }
  }

  // Accesses of one kind with the same steps and conditions, from bases that
  // differ by constants, are one wider access where their bytes form one
  // interval (the parts of a vectorized and interleaved access, say).
  SmallVector<bool, 16> Taken(Protos.size(), false);
  for (size_t a = 0; a < Protos.size(); ++a) {
    if (Taken[a])
      continue;
    Proto &PA = Protos[a];
    bool W = isa<StoreInst>(PA.I);
    // Members as (offset from PA's base, size, index).
    SmallVector<std::tuple<int64_t, int64_t, size_t>, 8> Ms;
    for (size_t b = a; b < Protos.size(); ++b) {
      Proto &PB = Protos[b];
      if (Taken[b] || isa<StoreInst>(PB.I) != W || PB.Dims != PA.Dims ||
          PB.Preds != PA.Preds || PB.Depth != PA.Depth ||
          PB.Partly != PA.Partly)
        continue;
      std::optional<int64_t> Off = constOffset(SE, PB.Base, PA.Base);
      if (!Off)
        continue;
      Taken[b] = true;
      Ms.push_back({*Off, PB.Size, b});
    }
    llvm::sort(Ms);
    // Each maximal run of touching intervals is one access.
    for (size_t r = 0; r < Ms.size();) {
      int64_t Lo = std::get<0>(Ms[r]), Hi = Lo + std::get<1>(Ms[r]);
      size_t e = r + 1;
      while (e < Ms.size() && std::get<0>(Ms[e]) <= Hi) {
        Hi = std::max(Hi, std::get<0>(Ms[e]) + std::get<1>(Ms[e]));
        ++e;
      }
      SmallVector<Instruction *, 4> Members;
      for (size_t k = r; k < e; ++k)
        Members.push_back(Protos[std::get<2>(Ms[k])].I);
      r = e;
      // Widen [0, Len) by every dimension whose step is at most Len. MinLen
      // is a constant lower bound on Len (every count is at least 1).
      const SCEV *Len = SE.getConstant(I64, Hi - Lo);
      int64_t MinLen = Hi - Lo;
      SmallVector<std::pair<const SCEV *, const SCEV *>, 4> Rest(PA.Dims);
      for (bool Grew = true; Grew;) {
        Grew = false;
        for (auto *It = Rest.begin(); It != Rest.end(); ++It) {
          auto [Step, N] = *It;
          const auto *C = dyn_cast<SCEVConstant>(Step);
          bool Fits = C ? !C->getAPInt().isNegative() &&
                              C->getAPInt().getSExtValue() <= MinLen
                        : SE.getMinusSCEV(Step, Len)->isZero();
          if (!Fits)
            continue;
          Len = SE.getAddExpr(
              Len, SE.getMulExpr(Step, SE.getMinusSCEV(N, SE.getOne(I64))));
          if (C)
            if (const auto *NC = dyn_cast<SCEVConstant>(N))
              MinLen += C->getAPInt().getSExtValue() *
                        (NC->getAPInt().getSExtValue() - 1);
          Rest.erase(It);
          Grew = true;
          break;
        }
      }
      // Worth it only if a dimension was dropped or merged, with at most two
      // left to loop over.
      if (Rest.size() == PA.Dims.size() && PA.Dims.size() == PA.Depth) {
        LLVM_DEBUG(dbgs() << "  footprint doesn't shrink: " << *Members[0]
                          << "\n");
        continue;
      }
      if (Rest.size() > 2) {
        LLVM_DEBUG(dbgs() << "  more than two rows: " << *Members[0] << "\n");
        continue;
      }
      // Outermost row (largest step) first.
      std::reverse(Rest.begin(), Rest.end());
      const SCEV *Base = SE.getAddExpr(PA.Base, SE.getConstant(I64, Lo, true));
      Prints.push_back({Members[0], Base, Len, {Rest.begin(), Rest.end()},
                        PA.Preds, std::move(Members), PA.Partly});
    }
  }
  if (Prints.empty())
    return Reject("no access to cover");

  // A store's checks cover a load with the same footprint.
  auto Same = [](const SymFootprint &A, const SymFootprint &B) {
    return A.Base == B.Base && A.Len == B.Len && A.Rows == B.Rows &&
           A.Preds == B.Preds && A.Partly == B.Partly;
  };
  for (size_t a = 0; a < Prints.size(); ++a)
    for (size_t b = 0; b < Prints.size() && !Prints[a].Dup; ++b)
      if (b != a && Same(Prints[a], Prints[b]) &&
          (isa<StoreInst>(Prints[b].I) > isa<StoreInst>(Prints[a].I) ||
           (isa<StoreInst>(Prints[b].I) == isa<StoreInst>(Prints[a].I) &&
            b < a)))
        Prints[a].Dup = true;
  // A check also stands for every footprint the same as its own.
  for (SymFootprint &P : Prints) {
    if (P.Dup)
      continue;
    SmallSetVector<Instruction *, 4> Members;
    for (SymFootprint &Q : Prints)
      if (Same(P, Q))
        Members.insert(Q.Members.begin(), Q.Members.end());
    P.Members.assign(Members.begin(), Members.end());
  }
  return true;
}

// The function footprint markers call (see placeFootprintMarkers).
static FunctionCallee getFootprintMarker(Module &M) {
  LLVMContext &C = M.getContext();
  FunctionType *FnTy = FunctionType::get(
      Type::getVoidTy(C),
      {Type::getInt1Ty(C), PointerType::getUnqual(C), Type::getInt64Ty(C)},
      /*isVarArg*/ true);
  AttributeList AL;
  AL = AL.addFnAttribute(C, Attribute::NoUnwind);
  AL = AL.addFnAttribute(C, Attribute::WillReturn);
  AL = AL.addFnAttribute(
      C, Attribute::getWithMemoryEffects(C, MemoryEffects::inaccessibleMemOnly()));
  AL = AL.addParamAttribute(
      C, 1, Attribute::getWithCaptureInfo(C, CaptureInfo::none()));
  return M.getOrInsertFunction(FootprintMarkerName, FnTy, AL);
}

static bool isFootprintMarker(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *Fn = CI->getCalledFunction())
      return Fn->getName() == FootprintMarkerName;
  return false;
}

// Footprint markers (see HoistSymbolicNests). Before the vectorizer versions,
// widens and unrolls the loops, the footprints of each symbolic nest in F
// (see symbolicFootprints) are expanded in its preheader, and each is passed
// to a marker,
//   call void @__cilksan_fp(i1 %guard, ptr %base, i64 %len,
//                           [i64 %step, i64 %count]...), !cilksan.fp !N
// where !N is a node distinct to the marker, and the accesses it covers get
// the same tag. The marker only touches memory nothing else can see, so it
// stays where it is, and the tag stays on the accesses, or is dropped (the
// vectorizer keeps a tag every access it merges has; see
// getMetadataToPropagate). Cilksan takes the markers out before instrumenting
// (see takeFootprintMarkers). An access is tagged only if its loop has one
// version and the footprint covers it fully, so the marker stands for it on
// every path; the others are checked as before. Returns the number of markers.
unsigned CilkSanitizerImpl::placeFootprintMarkers(Function &F, LoopInfo &LI,
                                                  ScalarEvolution &SE,
                                                  DominatorTree &DT,
                                                  TaskInfo &TI) {
  Module &M = *F.getParent();
  LLVMContext &C = F.getContext();
  Type *I64 = Type::getInt64Ty(C);
  unsigned Placed = 0;
  // Outermost first: the subloops of a nest with markers are not tried on
  // their own.
  SmallVector<Loop *, 8> Work(LI.begin(), LI.end());
  while (!Work.empty()) {
    Loop *O = Work.pop_back_val();
    SmallVector<SymFootprint, 8> Prints;
    DenseMap<Instruction *, unsigned> NumVersions;
    if (!symbolicFootprints(
            O, LI, SE, DT, TI,
            [&SE](Instruction &I) { return plainAccessSize(I, SE); }, Prints,
            NumVersions)) {
      Work.append(O->begin(), O->end());
      continue;
    }
    DenseMap<Instruction *, unsigned> NumChecks;
    for (const SymFootprint &P : Prints)
      if (!P.Dup)
        for (Instruction *I : P.Members)
          ++NumChecks[I];
    Instruction *Term = O->getLoopPreheader()->getTerminator();
    SCEVExpander Expander(SE, M.getDataLayout(), "cilksan.fp");
    struct Marker {
      SmallVector<Value *, 7> Args;
      SmallVector<Instruction *, 4> Tagged;
      DebugLoc DL;
    };
    SmallVector<Marker, 4> Markers;
    for (const SymFootprint &P : Prints) {
      if (P.Dup || P.Partly)
        continue;
      Marker K;
      for (Instruction *I : P.Members)
        if (NumVersions.lookup(I) == 1 && NumChecks.lookup(I) == 1)
          K.Tagged.push_back(I);
      if (K.Tagged.empty())
        continue;
      IRBuilder<> IRB(Term);
      Value *Guard = IRB.getTrue();
      for (auto [Cond, Dir] : P.Preds)
        Guard = IRB.CreateAnd(Guard, Dir ? Cond : IRB.CreateNot(Cond));
      K.Args = {Guard,
                Expander.expandCodeFor(P.Base, PointerType::getUnqual(C), Term),
                Expander.expandCodeFor(P.Len, I64, Term)};
      for (auto [Step, N] : P.Rows) {
        K.Args.push_back(Expander.expandCodeFor(Step, I64, Term));
        K.Args.push_back(Expander.expandCodeFor(N, I64, Term));
      }
      K.DL = P.I->getDebugLoc();
      Markers.push_back(std::move(K));
    }
    if (Markers.empty()) {
      Work.append(O->begin(), O->end());
      continue;
    }
    // The markers go together at the end of the preheader, after the values
    // they're called with.
    FunctionCallee Fn = getFootprintMarker(M);
    for (Marker &K : Markers) {
      CallInst *Call = CallInst::Create(Fn, K.Args, "", Term->getIterator());
      Call->setDebugLoc(K.DL);
      MDNode *Tag = MDNode::getDistinct(C, {});
      Call->setMetadata(FootprintTagKind, Tag);
      for (Instruction *I : K.Tagged)
        I->setMetadata(FootprintTagKind, Tag);
      LLVM_DEBUG(dbgs() << "Footprint marker for " << K.Tagged.size()
                        << " accesses before loop nest "
                        << O->getHeader()->getName() << ": " << *Call << "\n");
    }
    Placed += Markers.size();
  }
  return Placed;
}

// Whether marker K and access M, which K dominates, run in one strand: they
// are in one task, and no path from K to M passes a call (other than a
// memory-free intrinsic, a call that can't race or another marker), detach,
// reattach or sync. Nothing between VectorizerStart and Cilksan should break
// the strand the early pass found; this checks it.
static bool oneStrandFromMarker(Instruction *K, Instruction *M, TaskInfo &TI,
                                function_ref<bool(const Instruction &)> CannotRace) {
  BasicBlock *KB = K->getParent(), *MB = M->getParent();
  if (TI.getTaskFor(KB) != TI.getTaskFor(MB))
    return false;
  auto Ends = [&](const Instruction &I) {
    if (isa<DetachInst>(I) || isa<ReattachInst>(I) || isa<SyncInst>(I))
      return true;
    return isa<CallBase>(I) && !isa<DbgInfoIntrinsic>(I) &&
           !(isa<IntrinsicInst>(I) && !I.mayReadOrWriteMemory()) &&
           !CannotRace(I) && !isFootprintMarker(I);
  };
  // Whether the instructions in [From, To) let the strand go on.
  auto Clear = [&](BasicBlock::iterator From, BasicBlock::iterator To) {
    for (; From != To; ++From)
      if (Ends(*From))
        return false;
    return true;
  };
  if (KB == MB && K->comesBefore(M))
    return Clear(std::next(K->getIterator()), M->getIterator());
  if (!Clear(MB->begin(), M->getIterator()))
    return false;
  SmallPtrSet<BasicBlock *, 16> Seen;
  SmallVector<BasicBlock *, 16> Work(pred_begin(MB), pred_end(MB));
  while (!Work.empty()) {
    BasicBlock *B = Work.pop_back_val();
    if (!Seen.insert(B).second)
      continue;
    if (Seen.size() > 512)
      return false;
    if (B == KB) {
      if (!Clear(std::next(K->getIterator()), B->end()))
        return false;
      continue;
    }
    // M's own block again, on a later iteration: all of it runs.
    if (!Clear(B->begin(), B->end()))
      return false;
    Work.append(pred_begin(B), pred_end(B));
  }
  return true;
}

// Takes the footprint markers out of F (see placeFootprintMarkers), and pairs
// each tagged access with the nearest marker with its tag that dominates it,
// if they run in one strand (see oneStrandFromMarker). An access without one
// is checked on its own. The markers' values stay until the end, for their
// checks (see planFootprintChecks), and are deleted if unused then.
void CilkSanitizerImpl::takeFootprintMarkers(Function &F, DominatorTree &DT,
                                             TaskInfo &TI) {
  SmallVector<CallInst *, 8> Calls;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (isFootprintMarker(I))
        Calls.push_back(cast<CallInst>(&I));
  if (Calls.empty())
    return;
  SmallVector<FootprintMarker, 4> Markers;
  DenseMap<MDNode *, SmallVector<unsigned, 2>> ByTag;
  for (CallInst *Call : Calls) {
    Instruction *At = Call->getNextNode();
    while (isFootprintMarker(*At))
      At = At->getNextNode();
    FootprintMarker K{At, Call->getArgOperand(0), Call->getArgOperand(1),
                      Call->getArgOperand(2), {}, {}};
    for (unsigned a = 3; a + 1 < Call->arg_size(); a += 2)
      K.Rows.push_back({Call->getArgOperand(a), Call->getArgOperand(a + 1)});
    for (Value *V : Call->args())
      FootprintOperands.push_back(V);
    if (MDNode *Tag = Call->getMetadata(FootprintTagKind))
      ByTag[Tag].push_back(Markers.size());
    Markers.push_back(std::move(K));
  }
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      if (!isa<LoadInst>(I) && !isa<StoreInst>(I))
        continue;
      MDNode *Tag = I.getMetadata(FootprintTagKind);
      auto It = Tag ? ByTag.find(Tag) : ByTag.end();
      if (It == ByTag.end())
        continue;
      // The markers that dominate I dominate each other; take the last.
      int Best = -1;
      for (unsigned k : It->second)
        if (DT.dominates(Calls[k], &I) &&
            (Best < 0 || DT.dominates(Calls[Best], Calls[k])))
          Best = k;
      if (Best >= 0 &&
          oneStrandFromMarker(Calls[Best], &I, TI, simpleCallCannotRace)) {
        Markers[Best].Members.push_back(&I);
        ++NumFootprintTagsPaired;
      } else {
        LLVM_DEBUG(dbgs() << "Footprint marker lost for " << I << "\n");
        ++NumFootprintTagsLost;
      }
    }
  for (CallInst *Call : Calls)
    Call->eraseFromParent();
  auto &Kept = FootprintMarkers[&F];
  for (FootprintMarker &K : Markers)
    if (!K.Members.empty())
      Kept.push_back(std::move(K));
}

// Range checks for the footprint markers taken out of F, on the accesses N
// (see HoistNests) didn't cover. A marker's check is a write if an access it
// covers is a store (its footprint is the marker's, see symbolicFootprints,
// so the check finds what the store's checks would), and stands for the
// accesses that would be checked through the delayed path (see
// hoistableAccessSize). A marker without any is dropped.
void CilkSanitizerImpl::planFootprintChecks(
    Function &F, ScalarEvolution &SE, RaceInfo &RI,
    SmallPtrSetImpl<Instruction *> &Covered) {
  auto It = FootprintMarkers.find(&F);
  if (It == FootprintMarkers.end())
    return;
  for (FootprintMarker &K : It->second) {
    Value *GuardV = K.Guard;
    auto *Guard = dyn_cast_or_null<ConstantInt>(GuardV);
    if (!GuardV || !K.Base || !K.Len || (Guard && Guard->isZero()))
      continue;
    SmallVector<Instruction *, 8> Need;
    Instruction *Store = nullptr;
    for (Instruction *I : K.Members) {
      if (Covered.count(I))
        continue;
      if (!Store && isa<StoreInst>(I))
        Store = I;
      if (hoistableAccessSize(*I, RI, SE))
        Need.push_back(I);
    }
    if (Need.empty())
      continue;
    Instruction *Rep = nullptr;
    for (Instruction *I : Need)
      if (isa<StoreInst>(I)) {
        Rep = I;
        break;
      }
    if (!Rep)
      Rep = Store ? Store : Need[0];
    NestCheck NC{Rep, nullptr, nullptr, 0, K.At};
    NC.AddrV = K.Base;
    NC.LenV = K.Len;
    for (auto &[Step, N] : K.Rows) {
      if (!Step || !N)
        break;
      NC.RowsV.push_back({Step, N});
    }
    if (NC.RowsV.size() != K.Rows.size())
      continue;
    if (!Guard)
      NC.Preds.push_back({GuardV, true});
    NC.Members.append(Need.begin(), Need.end());
    Covered.insert(Need.begin(), Need.end());
    NestChecks.push_back(std::move(NC));
    ++NumFootprintMarkersUsed;
    ++NumSymNestChecks;
    NumSymNestAccesses += Need.size();
    LLVM_DEBUG(dbgs() << "Footprint check for " << Need.size()
                      << " accesses before " << *K.At << "\n");
  }
}

// Removes what footprint markers leave behind: their values, if no check
// uses them, the tags and the attribute, in every function, and the marker's
// declaration.
void CilkSanitizerImpl::removeFootprintMarkers() {
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(FootprintOperands);
  FootprintOperands.clear();
  FootprintMarkers.clear();
  for (Function &F : M)
    F.removeFnAttr(FootprintsAttr);
  // Tags exist only if a marker was placed, which declared it.
  Function *Fn = M.getFunction(FootprintMarkerName);
  if (!Fn)
    return;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        I.setMetadata(FootprintTagKind, nullptr);
  if (Fn->use_empty())
    Fn->eraseFromParent();
}

// Straight-line accesses checked as a whole (see CoalesceBlocks). Between
// two calls in a basic block, the accesses run in one strand, in order, all or
// none, so as for a loop nest, the exact bytes each base is read and written
// at can be checked in any order: one check per run of written bytes, and of
// bytes read but not written. The checks go right after the last access they
// cover, where every address is computed. Calls (other than memory-free
// intrinsics), atomics and fences end a stretch, so no other instrumented
// event moves across a check, and so does any join or branch.
void CilkSanitizerImpl::planBlockChecks(
    Function &F, ScalarEvolution &SE, RaceInfo &RI,
    SmallVectorImpl<Instruction *> &LoadsAndStores) {
  // Accesses checked through the delayed path, with their sizes.
  SmallDenseMap<Instruction *, int64_t, 32> Cands;
  for (Instruction *I : LoadsAndStores)
    if (std::optional<int64_t> Size = hoistableAccessSize(*I, RI, SE))
      Cands[I] = *Size;
  LLVM_DEBUG(dbgs() << "Straight-line candidates in " << F.getName() << ": "
                    << Cands.size() << " of " << LoadsAndStores.size() << "\n");
  if (Cands.size() < 2)
    return;

  struct Class {
    Value *BasePtr;     // the first access's pointer
    const SCEV *BaseEv; // and its SCEV
    SmallVector<FootprintAccess, 8> Accesses; // in program order
  };
  SmallPtrSet<Instruction *, 32> Covered;
  SmallVector<Class, 4> Classes;
  auto Flush = [&]() {
    for (Class &K : Classes) {
      if (K.Accesses.size() < 2)
        continue;
      // The checks go after the last access, reporting the first of each
      // kind, and are only worth it with fewer checks than accesses. The base
      // is an opaque value: expanding its SCEV inside a loop could add an
      // induction variable.
      Instruction *Last = K.Accesses.back().I;
      unsigned Checks = coverClass(
          K.Accesses, SE, SE.getUnknown(K.BasePtr), /*O=*/nullptr,
          Last->getNextNode(), /*FirstRep=*/true, /*MaxSpan=*/4096,
          /*MaxMarks=*/UINT64_MAX, /*MaxRuns=*/K.Accesses.size() - 1);
      if (!Checks)
        continue;
      for (FootprintAccess &A : K.Accesses)
        Covered.insert(A.I);
      NumBlockChecks += Checks;
      NumBlockAccesses += K.Accesses.size();
      LLVM_DEBUG(dbgs() << "Checking " << K.Accesses.size()
                        << " straight-line accesses with " << Checks
                        << " range checks after " << *Last << "\n");
    }
    Classes.clear();
  };

  // A stretch runs on through a chain of blocks, each the only successor of
  // the one before and with it as only predecessor: instrumentation setup
  // splits blocks at calls, even to intrinsics such as fmuladd.
  BasicBlock *Prev = nullptr;
  for (BasicBlock &BB : F) {
    if (!(Prev && BB.getSinglePredecessor() == Prev &&
          Prev->getSingleSuccessor() == &BB))
      Flush();
    Prev = &BB;
    for (Instruction &I : BB) {
      bool EndsStretch =
          isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) ||
          isa<FenceInst>(I) ||
          (isa<CallBase>(I) && !isa<DbgInfoIntrinsic>(I) &&
           !(isa<IntrinsicInst>(I) && !I.mayReadOrWriteMemory()));
      if (auto *Ld = dyn_cast<LoadInst>(&I))
        EndsStretch |= Ld->isVolatile() || !Ld->isUnordered();
      if (auto *St = dyn_cast<StoreInst>(&I))
        EndsStretch |= St->isVolatile() || !St->isUnordered();
      if (EndsStretch) {
        Flush();
        continue;
      }
      auto CIt = Cands.find(&I);
      if (CIt == Cands.end())
        continue;
      Value *Ptr = getLoadStorePointerOperand(&I);
      const SCEV *V = SE.getSCEV(Ptr);
      int64_t Size = CIt->second;
      Class *C = nullptr;
      int64_t Off = 0;
      for (Class &K : Classes)
        if (std::optional<int64_t> D = constOffset(SE, V, K.BaseEv)) {
          Off = *D;
          C = &K;
          break;
        }
      if (!C) {
        Classes.push_back({Ptr, V, {}});
        C = &Classes.back();
      }
      C->Accesses.push_back({&I, Off, Size});
    }
  }
  Flush();

  llvm::erase_if(LoadsAndStores,
                 [&](Instruction *I) { return Covered.count(I); });
}

bool CilkSanitizerImpl::instrumentFunctionUsingRI(Function &F) {

  if (F.empty() || shouldNotInstrumentFunction(F) ||
      !checkSanitizeCilkAttr(F)) {
    LLVM_DEBUG({
        dbgs() << "Skipping " << F.getName() << "\n";
        if (F.empty())
          dbgs() << "  Empty function\n";
        else if (shouldNotInstrumentFunction(F))
          dbgs() << "  Function should not be instrumented\n";
        else if (!checkSanitizeCilkAttr(F))
          dbgs() << "  Function lacks sanitize_cilk attribute\n";});
    return false;
  }

  LLVM_DEBUG(dbgs() << "Instrumenting " << F.getName() << "\n");

  SmallVector<Instruction *, 8> AllLoadsAndStores;
  SmallVector<Instruction *, 8> LocalLoadsAndStores;
  SmallVector<Instruction *, 8> AtomicAccesses;
  SmallVector<Instruction *, 8> MemIntrinCalls;
  SmallVector<Instruction *, 8> IntrinsicCalls;
  SmallVector<Instruction *, 8> LibCalls;
  SmallVector<Instruction *, 8> Callsites;
  // Ancillary instructions
  SmallPtrSet<Instruction *, 8> Allocas;
  SmallPtrSet<Instruction *, 8> AllocationFnCalls;
  SmallPtrSet<Instruction *, 8> FreeCalls;
  SmallVector<SyncInst *, 8> Syncs;
  DenseMap<BasicBlock *, unsigned> SRCounters;
  DenseMap<Value *, unsigned> SyncRegNums;

  // Find instructions that can be hoisted or sinked
  SmallPtrSet<Instruction *, 8> LoopInstToHoist;
  SmallPtrSet<Instruction *, 8> LoopInstToSink;
  SmallPtrSet<const Loop *, 8> TapirLoops;

  const TargetLibraryInfo *TLI = &GetTLI(F);
  DominatorTree &DT = GetDomTree(F);
  LoopInfo &LI = GetLoopInfo(F);
  TaskInfo &TI = GetTaskInfo(F);
  RaceInfo &RI = GetRaceInfo(F);

  // Reset the per-function label-hoisting state before any instrumentation.
  CurTaskInfo = &TI;
  CurDomTree = &DT;
  CurLoopInfo = &LI;
  SpindleLabelCache.clear();
  NestChecks.clear();

  ICFLoopSafetyInfo SafetyInfo;

  ScalarEvolution &SE = *(RI.getSE());

  // Accesses whose checks are hoisted before their loop nest.
  SmallPtrSet<Instruction *, 32> NestCovered;
  if (EnableStaticRaceDetection && LoopHoisting && HoistNests)
    planNestChecks(LI, SE, DT, TI, RI, NestCovered);
  if (EnableStaticRaceDetection && LoopHoisting && HoistNests &&
      HoistSymbolicNests)
    planFootprintChecks(F, SE, RI, NestCovered);

  for (BasicBlock &BB : F) {
    // Record the Tapir sync instructions found
    if (SyncInst *SI = dyn_cast<SyncInst>(BB.getTerminator()))
      Syncs.push_back(SI);

    // get loop for BB
    Loop *L = LI.getLoopFor(&BB);
    if (L)
      SafetyInfo.computeLoopSafetyInfo(L);

    // Record the memory accesses in the basic block
    for (Instruction &Inst : BB) {
      bool CanCoalesce = NestCovered.count(&Inst);
      // If the instruction is in a loop and can only race via ancestor, and
      // size < stride, store it.
      if (!CanCoalesce && L && EnableStaticRaceDetection && LoopHoisting &&
          SafetyInfo.isGuaranteedToExecute(Inst, &DT, &TI, L)) {
        // TODO: For now, only look at loads and stores.  Add atomics later.
        //       Need to add any others?
        if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst)) {
          bool RaceViaAncestor = false;
          bool OtherRace = false;
          for (const RaceInfo::RaceData &RD : RI.getRaceData(&Inst)) {
            if (RaceInfo::isRaceViaAncestor(RD.Type)) {
              RaceViaAncestor = true;
            } else if (RaceInfo::isOpaqueRace(RD.Type)) {
              LLVM_DEBUG(dbgs() << "Can't hoist or sink instrumentation for "
                                << Inst << "\n  Opaque race.\n");
              OtherRace = true;
              break;
            } else if (RaceInfo::isLocalRace(RD.Type)) {
              if (!RD.Racer.isValid()) {
                LLVM_DEBUG(dbgs()
                           << "Can't hoist or sink instrumentation for " << Inst
                           << "\n  Local race with opaque racer.\n");
                OtherRace = true;
                break;
              }
              if (LI.getLoopFor(RD.Racer.I->getParent()) == L) {
                LLVM_DEBUG(dbgs()
                           << "Can't hoist or sink instrumentation for " << Inst
                           << "\n  Local race with racer in same loop: "
                           << *RD.Racer.I << "\n");
                OtherRace = true;
                break;
              }
              RaceViaAncestor = true;
            }
          }
          // If this instruction can only race via an ancestor, see if it can be
          // hoisted.
          if (RaceViaAncestor && !OtherRace) {
            const SCEV *Size = accessSize(SE, &Inst);
            const SCEV *V = SE.getSCEV(getLoadStorePointerOperand(&Inst));
            // If not an AddRecExpr, don't proceed
            if (const SCEVAddRecExpr *SrcAR = dyn_cast<SCEVAddRecExpr>(V)) {
              const SCEV *Stride = SrcAR->getStepRecurrence(SE);
              const SCEV *Diff;
              if (SE.isKnownNonNegative(Stride)) {
                Diff = SE.getMinusSCEV(Size, Stride);
              } else {
                // If we can't compare size and stride,
                // SE.isKnownNonNegative(Diff) will be false.
                Diff = SE.getAddExpr(Size, Stride);
              }
              bool IsTapirLoop = static_cast<bool>(getTaskIfTapirLoop(L, &TI));
              if (IsTapirLoop)
                TapirLoops.insert(L);
              const SCEV *TripCount = getRuntimeTripCount(*L, &SE, IsTapirLoop);

              if (SE.isKnownNonNegative(Diff)) {
                if (!isa<SCEVCouldNotCompute>(TripCount) &&
                    SE.isAvailableAtLoopEntry(SrcAR->getStart(), L)) {
                  // Can hoist if stride <= size and the tripcount is known and
                  // the start is available at loop entry.
                  LoopInstToHoist.insert(&Inst);
                  CanCoalesce = true;
                  LLVM_DEBUG(dbgs() << "Can hoist instrumentation for " << Inst << "\n");
                } else if (!isa<SCEVCouldNotCompute>(
                               SE.getConstantMaxBackedgeTakenCount(L))) {
                  // Can sink if stride <= size and the tripcount is unknown but
                  // guaranteed to be finite.
                  LoopInstToSink.insert(&Inst);
                  CanCoalesce = true;
                  LLVM_DEBUG(dbgs() << "Can sink instrumentation for " << Inst << "\n");
                } else {
                  LLVM_DEBUG(dbgs()
                             << "Can't hoist or sink instrumentation for "
                             << Inst << "\n  TripCount = " << *TripCount
                             << "\n  SrcAR->getStart() = " << *SrcAR->getStart()
                             << "\n  SE.getConstantMaxBackedgeTakenCount(L) = "
                             << *SE.getConstantMaxBackedgeTakenCount(L)
                             << "\n");
                }
              } else {
                LLVM_DEBUG(dbgs() << "Can't hoist instrumentation for " << Inst
                                  << "\n  Diff SCEV not known non-negative: "
                                  << *Diff << "\n");
              }
            } else {
              LLVM_DEBUG(
                  dbgs()
                  << "Can't hoist or sink instrumentation for " << Inst
                  << "\n  SCEV for load/store pointer operand not AddRecExpr: "
                  << *V << ": " << V->getSCEVType() << "\n");
            }
          }
        }
      }

      if (!CanCoalesce) {
        // TODO: Handle VAArgInst
        if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
          LocalLoadsAndStores.push_back(&Inst);
        else if (isa<AtomicRMWInst>(Inst) || isa<AtomicCmpXchgInst>(Inst))
          AtomicAccesses.push_back(&Inst);
        else if (isa<AllocaInst>(Inst))
          Allocas.insert(&Inst);
        else if (isa<CallBase>(Inst)) {
          // if (CallInst *CI = dyn_cast<CallInst>(&Inst))
          //   maybeMarkSanitizerLibraryCallNoBuiltin(CI, TLI);

          // If we find a sync region, record it.
          if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst))
            if (Intrinsic::syncregion_start == II->getIntrinsicID()) {
              // Identify this sync region with a counter value, where all sync
              // regions within a function or task are numbered from 0.
              if (TI.getTaskFor(&BB)) {
                BasicBlock *TEntry = TI.getTaskFor(&BB)->getEntry();
                // Create a new counter if need be.
                if (!SRCounters.count(TEntry))
                  SRCounters[TEntry] = 0;
                SyncRegNums[&Inst] = SRCounters[TEntry]++;
              }
            }

          // Record this function call as either an allocation function, a call to
          // free (or delete), a memory intrinsic, or an ordinary real function
          // call.
          if (isAllocFn(&Inst, TLI))
            AllocationFnCalls.insert(&Inst);
          else if (isFreeFn(&Inst, TLI))
            FreeCalls.insert(&Inst);
          else if (isa<AnyMemIntrinsic>(Inst))
            MemIntrinCalls.push_back(&Inst);
          else if (!simpleCallCannotRace(Inst) && !shouldIgnoreCall(Inst)) {
            if (isa<IntrinsicInst>(&Inst)) {
              if (Inst.mayReadOrWriteMemory())
                IntrinsicCalls.push_back(&Inst);
            } else if (isLibCall(Inst, TLI)) {
              if (Inst.mayReadOrWriteMemory())
                LibCalls.push_back(&Inst);
            } else {
              Callsites.push_back(&Inst);
            }
          }
        }

        // Add the current set of local loads and stores to be considered for
        // instrumentation.
        if (!simpleCallCannotRace(Inst)) {
          chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores,
                                         TI, LI, TLI);
        }
      }
    }
    chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores, TI,
                                   LI, TLI);
  }

  if (EnableStaticRaceDetection && CoalesceBlocks)
    planBlockChecks(F, SE, RI, AllLoadsAndStores);

  // Evaluate the tasks that might be in parallel with each spindle.
  MaybeParallelTasks MPTasks;
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);

  // Map each detach instruction with the sync instructions that could sync it.
  for (SyncInst *Sync : Syncs)
    for (const Task *MPT :
             MPTasks.TaskList[TI.getSpindleFor(Sync->getParent())])
      DetachToSync[MPT->getDetach()].push_back(Sync);

  // Record objects involved in races
  for (auto &ObjRD : RI.getObjectMRForRace())
    ObjectMRForRace[ObjRD.first] = ObjRD.second;

  uint64_t LocalId = getLocalFunctionID(F);
  IRBuilder<> IRB(getEntryBBInsertPt(F.getEntryBlock()));
  Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);

  bool Result = false;
  if (!EnableStaticRaceDetection) {
    SimpleInstrumentor FuncI(*this, TI, LI, DT, TLI);
    Result |= FuncI.instrumentSimpleInstructions(AllLoadsAndStores);
    Result |= FuncI.instrumentSimpleInstructions(AtomicAccesses);
    Result |= FuncI.instrumentAnyMemIntrinsics(MemIntrinCalls);
    Result |= FuncI.instrumentCalls(IntrinsicCalls);
    Result |= FuncI.instrumentCalls(LibCalls);
    Result |= FuncI.instrumentCalls(Callsites);

    // Instrument ancillary instructions including allocas, allocation-function
    // calls, free calls, detaches, and syncs.
    Result |= FuncI.instrumentAncillaryInstructions(Allocas, AllocationFnCalls,
                                                    FreeCalls, SyncRegNums,
                                                    SRCounters, DL);
  } else {
    Instrumentor FuncI(*this, RI, TI, LI, DT, TLI);

    // Insert MAAP flags for each function argument.
    FuncI.insertArgMAAPs(F, FuncId);

    Result |= FuncI.instrumentSimpleInstructions(AllLoadsAndStores);
    Result |= FuncI.instrumentSimpleInstructions(AtomicAccesses);
    Result |= FuncI.instrumentAnyMemIntrinsics(MemIntrinCalls);
    Result |= FuncI.instrumentCalls(IntrinsicCalls);
    Result |= FuncI.instrumentCalls(LibCalls);
    Result |= FuncI.instrumentCalls(Callsites);

    // Find detaches that need to be instrumented for loop instructions whose
    // instrumentation will be coalesced.
    FuncI.getDetachesForCoalescedInstrumentation(LoopInstToHoist,
                                                 LoopInstToSink);

    // Instrument ancillary instructions including allocas, allocation-function
    // calls, free calls, detaches, and syncs.
    Result |= FuncI.instrumentAncillaryInstructions(Allocas, AllocationFnCalls,
                                                    FreeCalls, SyncRegNums,
                                                    SRCounters, DL);

    // Hoist and sink instrumentation when possible (applies to all loops,
    // not just Tapir loops)
    // Also inserts MAAP checks for hoisted/sinked instrumentation
    Result |=
        FuncI.instrumentLoops(LoopInstToHoist, LoopInstToSink, TapirLoops, &SE);

    // Once we have handled ancillary instructions, we've done the necessary
    // analysis on this function.  We now perform delayed instrumentation, which
    // can involve changing the CFG and thereby violating some analyses.
    Result |= FuncI.performDelayedInstrumentation();
  }

  if (Result) {
    bool MaySpawn = !TI.isSerial();
    if (InstrumentationSet & SERIESPARALLEL) {
      IRBuilder<> IRB(cast<Instruction>(FuncId)->getNextNode());
      CsiFuncProperty FuncEntryProp;
      FuncEntryProp.setMaySpawn(MaySpawn);
      if (MaySpawn)
        FuncEntryProp.setNumSyncReg(SRCounters[TI.getRootTask()->getEntry()]);
      // TODO: Determine if we actually want the frame pointer, not the stack
      // pointer.
      Value *FrameAddr =
          IRB.CreateCall(Intrinsic::getOrInsertDeclaration(
                             &M, Intrinsic::frameaddress, IRB.getPtrTy()),
                         {IRB.getInt32(0)});
      Value *StackSave = IRB.CreateStackSave();
      CallInst *EntryCall =
          IRB.CreateCall(CsanFuncEntry, {FuncId, FrameAddr, StackSave,
                                         FuncEntryProp.getValue(IRB)});
      setInstrumentationDebugLoc(F, EntryCall);
    } else {
      // Search for a call to CsanFuncEntry, and update its ID argument.
      for (BasicBlock::iterator I = cast<Instruction>(FuncId)->getIterator(),
                                E = F.getEntryBlock().end();
           I != E; ++I) {
        if (CallBase *CB = dyn_cast<CallBase>(&*I))
          if (CB->getCalledFunction() == CsanFuncEntry.getCallee()) {
            CB->setArgOperand(0, FuncId);
            break;
          }
      }
    }

    EscapeEnumerator EE(F, "csan_cleanup", false);
    while (IRBuilder<> *AtExit = EE.Next()) {
      if (InstrumentationSet & SERIESPARALLEL) {
        uint64_t ExitLocalId = FunctionExitFED.add(*AtExit->GetInsertPoint());
        Value *ExitCsiId = FunctionExitFED.localToGlobalId(ExitLocalId, *AtExit);
        CsiFuncExitProperty FuncExitProp;
        FuncExitProp.setMaySpawn(MaySpawn);
        FuncExitProp.setEHReturn(isa<ResumeInst>(AtExit->GetInsertPoint()));
        CallInst *ExitCall = AtExit->CreateCall(
            CsanFuncExit, {ExitCsiId, FuncId, FuncExitProp.getValue(*AtExit)});
        setInstrumentationDebugLoc(F, ExitCall);
      } else {
        // Search for a call to CsanFuncExit, and update its ID argument.
        for (BasicBlock::iterator I = AtExit->GetInsertBlock()->begin(),
                                  E = AtExit->GetInsertBlock()->end();
             I != E; ++I) {
          if (CallBase *CB = dyn_cast<CallBase>(&*I))
            if (CB->getCalledFunction() == CsanFuncExit.getCallee()) {
              CB->setArgOperand(1, FuncId);
              break;
            }
        }
      }
    }
  }

  // Record aggregate race information for the function and its arguments for
  // interprocedural analysis.
  //
  // TODO: Clean this up
  RaceInfo::RaceType FuncRT = RaceInfo::None;
  for (Instruction *I : AllLoadsAndStores)
    FuncRT = RaceInfo::unionRaceTypes(FuncRT, RI.getRaceType(I));
  for (Instruction *I : AtomicAccesses)
    FuncRT = RaceInfo::unionRaceTypes(FuncRT, RI.getRaceType(I));
  for (Instruction *I : MemIntrinCalls)
    FuncRT = RaceInfo::unionRaceTypes(FuncRT, RI.getRaceType(I));
  for (Instruction *I : Callsites) {
    if (const CallBase *CB = dyn_cast<CallBase>(I)) {
      // Use updated information about the race type of the call, if it's
      // available.
      const Function *CF = CB->getCalledFunction();
      if (FunctionRaceType.count(CF)) {
        FuncRT = RaceInfo::unionRaceTypes(FuncRT, FunctionRaceType[CF]);
        // Preserve the local-race marking if the callsite itself is involved in
        // a local race.
        if (RaceInfo::isLocalRace(RI.getRaceType(I)))
          FuncRT = RaceInfo::unionRaceTypes(FuncRT, RaceInfo::Local);
        continue;
      }
    }
    FuncRT = RaceInfo::unionRaceTypes(FuncRT, RI.getRaceType(I));
  }
  FunctionRaceType[&F] = FuncRT;

  return Result;
}

bool CilkSanitizerImpl::instrumentLoadOrStore(Instruction *I,
                                              IRBuilder<> &IRB) {
  bool IsWrite = isa<StoreInst>(*I);
  Value *Addr = IsWrite
      ? cast<StoreInst>(I)->getPointerOperand()
      : cast<LoadInst>(I)->getPointerOperand();
  Type *Ty =
      IsWrite ? cast<StoreInst>(I)->getValueOperand()->getType() : I->getType();

  // swifterror memory addresses are mem2reg promoted by instruction selection.
  // As such they cannot have regular uses like an instrumentation function and
  // it makes no sense to track them as memory.
  if (Addr->isSwiftError())
    return false;

  int NumBytesAccessed = getNumBytesAccessed(Ty, DL);
  if (-1 == NumBytesAccessed) {
    // Ignore accesses with bad sizes.
    NumAccessesWithBadSize++;
    return false;
  }

  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SHADOWMEMORY))
    return true;

  const Align Alignment = IsWrite
      ? cast<StoreInst>(I)->getAlign()
      : cast<LoadInst>(I)->getAlign();
  CsiLoadStoreProperty Prop;
  Prop.setAlignment(Alignment);
  Prop.setIsAtomic(I->isAtomic());
  Prop.setIsThreadLocal(isThreadLocalObject(lookupUnderlyingObject(Addr)));
  if (IsWrite) {
    // Instrument store
    uint64_t LocalId = StoreFED.add(*I);
    uint64_t StoreObjId = StoreObj.add(*I, lookupUnderlyingObject(Addr));
    assert(LocalId == StoreObjId &&
           "Store received different ID's in FED and object tables.");
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    SmallVector<Value *, 5> Args = {
        CsiId,
        IRB.CreatePointerCast(Addr, IRB.getPtrTy()),
        IRB.getInt32(NumBytesAccessed),
        Prop.getValue(IRB)};
    if (StoreTakesLabel) {
      CallInst *CurLabCall = getCurOSLabelCall(IRB);
      Args.push_back(CurLabCall);
    }
    Instruction *Call = IRB.CreateCall(CsanWrite, Args);
    IRB.SetInstDebugLocation(Call);
    NumInstrumentedWrites++;
  } else {
    // Instrument load
    uint64_t LocalId = LoadFED.add(*I);
    uint64_t LoadObjId = LoadObj.add(*I, lookupUnderlyingObject(Addr));
    assert(LocalId == LoadObjId &&
           "Load received different ID's in FED and object tables.");
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
    SmallVector<Value *, 5> Args = {
        CsiId,
        IRB.CreatePointerCast(Addr, IRB.getPtrTy()),
        IRB.getInt32(NumBytesAccessed),
        Prop.getValue(IRB)};
    if (LoadTakesLabel) {
      CallInst *CurLabCall = getCurOSLabelCall(IRB);
      Args.push_back(CurLabCall);
    }
    Instruction *Call = IRB.CreateCall(CsanRead, Args);
    IRB.SetInstDebugLocation(Call);
    NumInstrumentedReads++;
  }
  return true;
}

bool CilkSanitizerImpl::instrumentAtomic(Instruction *I, IRBuilder<> &IRB) {
  Value *Addr;
  Type *Ty;
  Align Alignment;
  if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    Addr = RMWI->getPointerOperand();
    Ty = RMWI->getValOperand()->getType();
    Alignment = RMWI->getAlign();
  } else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
    Addr = CASI->getPointerOperand();
    Ty = CASI->getNewValOperand()->getType();
    Alignment = CASI->getAlign();
  } else {
    return false;
  }

  int NumBytesAccessed = getNumBytesAccessed(Ty, DL);
  if (-1 == NumBytesAccessed) {
    // Ignore accesses with bad sizes.
    NumAccessesWithBadSize++;
    return false;
  }

  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SHADOWMEMORY))
    return true;

  CsiLoadStoreProperty Prop;
  Prop.setAlignment(Alignment);
  Prop.setIsAtomic(true);
  Prop.setIsThreadLocal(isThreadLocalObject(lookupUnderlyingObject(Addr)));
  uint64_t LocalId = StoreFED.add(*I);
  uint64_t StoreObjId = StoreObj.add(*I, lookupUnderlyingObject(Addr));
  assert(LocalId == StoreObjId &&
         "Store received different ID's in FED and object tables.");
  Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
  SmallVector<Value *, 5> Args = {
      CsiId,
      IRB.CreatePointerCast(Addr, IRB.getPtrTy()),
      IRB.getInt32(NumBytesAccessed),
      Prop.getValue(IRB)};
  if (StoreTakesLabel) {
    CallInst *CurLabCall = getCurOSLabelCall(IRB);
    Args.push_back(CurLabCall);
  }
  Instruction *Call = IRB.CreateCall(CsanWrite, Args);
  IRB.SetInstDebugLocation(Call);
  NumInstrumentedWrites++;
  return true;
}

FunctionCallee CilkSanitizerImpl::getOrInsertSynthesizedHook(StringRef Name,
                                                             FunctionType *T,
                                                             int ReturnParam,
                                                             AttributeList AL) {
  // If no bitcode file has been linked, then we cannot check if it contains a
  // particular library hook.  Simply return the hook.  If the Cilksan library
  // doesn't contain that hook, the linker will raise an error.
  if (!LinkedBitcode)
    return getHookFunction(Name, T, AL);

  // Check if the linked bitcode file contains the library hook.  If it does,
  // return that hook.
  if (FunctionsInBitcode.contains(std::string(Name)))
    return getHookFunction(Name, T, AL);

  // If the function is already present, just return it.
  if (Function *F = M.getFunction(Name))
    if (F->getFunctionType() == T)
      return M.getOrInsertFunction(Name, T);

  // We did not find the library hook in the linked bitcode file.  Synthesize a
  // default version of the hook that simply calls __csan_default_libhook.
  FunctionCallee NewHook = M.getOrInsertFunction(Name, T, AL);
  Function *NewHookFn = cast<Function>(NewHook.getCallee());
  NewHookFn->setOnlyAccessesInaccessibleMemOrArgMem();
  NewHookFn->setDoesNotThrow();
  LLVMContext &Ctx = M.getContext();
  assert((T->getReturnType() == Type::getVoidTy(Ctx) || ReturnParam >= 0) &&
         "Synthesizing hook with return value, but no return parameter "
         "specified.");
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", NewHookFn);
  // Return void or the parameter at the specified index.
  ReturnInst *RI =
      (T->getReturnType() == Type::getVoidTy(Ctx) || ReturnParam < 0)
          ? ReturnInst::Create(Ctx, Entry)
          : ReturnInst::Create(Ctx, NewHookFn->getArg(ReturnParam), Entry);
  IRBuilder<> IRB(RI);

  // Insert a call to the default library function hook
  Type *IDType = IRB.getInt64Ty();
  FunctionType *DefaultHookTy =
      FunctionType::get(IRB.getVoidTy(),
                        {/*call_id*/
                         IDType, /*func_id*/ IDType,
                         /*MAAP_count*/ IRB.getInt8Ty()},
                        /*isVarArg*/ false);
  FunctionCallee DefaultHook =
      M.getOrInsertFunction("__csan_default_libhook", DefaultHookTy);
  IRB.CreateCall(DefaultHook, {NewHookFn->getArg(0), NewHookFn->getArg(1),
                               NewHookFn->getArg(2)});
  return NewHook;
}

// Check if we need to spill a value of this type onto the stack to pass it to a
// hook.
static bool needToSpillType(const Type *T) {
  return T->isVectorTy() || T->isStructTy();
}

bool CilkSanitizerImpl::instrumentIntrinsicCall(
    Instruction *I, SmallVectorImpl<Value *> *MAAPVals) {
  assert(!callsPlaceholderFunction(*I) &&
         "instrumentIntrinsicCall called on placeholder function");

  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SERIESPARALLEL))
    return true;

  CallBase *CB = dyn_cast<CallBase>(I);
  if (!CB)
    return false;
  Function *Called = CB->getCalledFunction();

  IRBuilder<> IRB(I);
  LLVMContext &Ctx = IRB.getContext();
  uint64_t LocalId = CallsiteFED.add(*I);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = getCalleeFuncID(Called, IRB);
  assert(FuncId != NULL);

  Value *NumMVVal = IRB.getInt8(0);
  if (MAAPVals && !MAAPVals->empty()) {
    unsigned NumMV = MAAPVals->size();
    NumMVVal = IRB.getInt8(NumMV);
  }

  CsiCallProperty Prop;
  // TODO: Set appropriate property values for this intrinsic call
  Value *PropVal = Prop.getValue(IRB);

  // Since C/C++ does not like '.' characters in function names, convert '.' to
  // '_' in the hook name.
  SmallString<256> Buf;
  for (char C : Called->getName().bytes()) {
    if ('.' == C)
      Buf.push_back('_');
    else
      Buf.push_back(C);
  }
  Type *IDType = IRB.getInt64Ty();

  // If the intrinsic does not return, insert the hook before the intrinsic.
  if (CB->doesNotReturn()) {
    // Synthesize the before hook for this function.
    SmallVector<Type *, 8> BeforeHookParamTys(
        {IDType, /*callee func_id*/ IDType,
         /*Num MAAPVal*/ IRB.getInt8Ty(), CsiCallProperty::getType(Ctx)});
    SmallVector<Value *, 8> BeforeHookParamVals(
        {CallsiteId, FuncId, NumMVVal, PropVal});

    // Populate the BeforeHook parameters with the parameters of the
    // instrumented function itself.
    Value *SavedStack = nullptr;
    const DataLayout &DL = M.getDataLayout();
    for (Value *Arg : CB->args()) {
      Type *ArgTy = Arg->getType();
      if (!needToSpillType(ArgTy)) {
        // We can simply pass the argument directly to the hook.
        BeforeHookParamTys.push_back(ArgTy);
        BeforeHookParamVals.push_back(Arg);
        continue;
      }
      // We need to spill the argument onto the stack.

      // Save the stack pointer, if we haven't already
      if (!SavedStack)
        SavedStack = IRB.CreateStackSave();

      // Spill the argument onto the stack
      AllocaInst *ArgSpill = IRB.CreateAlloca(ArgTy);
      IRB.CreateAlignedStore(Arg, ArgSpill, DL.getStackAlignment());

      // Add the spilled argument
      BeforeHookParamTys.push_back(ArgSpill->getType());
      BeforeHookParamVals.push_back(ArgSpill);
    }
    FunctionType *BeforeHookTy = FunctionType::get(
        IRB.getVoidTy(), BeforeHookParamTys, Called->isVarArg());
    FunctionCallee BeforeIntrinCallHook = getOrInsertSynthesizedHook(
        ("__csan_" + Buf).str(), BeforeHookTy);

    // Insert the hook before the call
    insertHookCall(I, BeforeIntrinCallHook, BeforeHookParamVals);

    // If we previously saved the stack pointer, restore it
    if (SavedStack)
      IRB.CreateStackRestore(SavedStack);
    return true;
  }

  // Otherwise, insert the hook after the intrinsic.
  assert(!isa<InvokeInst>(I) &&
         "instrumentIntrinsicCall called on invoke instruction");

  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);

  // Synthesize the after hook for this function.
  SmallVector<Type *, 8> AfterHookParamTys({IDType, /*callee func_id*/ IDType,
                                            /*Num MAAPVal*/ IRB.getInt8Ty(),
                                            CsiCallProperty::getType(Ctx)});
  SmallVector<Value *, 8> AfterHookParamVals(
      {CallsiteId, FuncId, NumMVVal, PropVal});

  // Populate the AfterHook parameters with the parameters of the instrumented
  // function itself.
  int ReturnParam = -1;
  Value *SavedStack = nullptr;
  const DataLayout &DL = M.getDataLayout();
  if (!Called->getReturnType()->isVoidTy()) {
    ReturnParam = AfterHookParamVals.size();
    Type *RetTy = Called->getReturnType();
    if (!needToSpillType(RetTy)) {
      // We can simply pass the return value directly to the hook.
      AfterHookParamTys.push_back(RetTy);
      AfterHookParamVals.push_back(CB);
    } else {
      // We need to spill the return value onto the stack.

      // Save the stack pointer, if we haven't already
      if (!SavedStack)
        SavedStack = IRB.CreateStackSave();

      // Spill the return value onto the stack
      AllocaInst *RetSpill = IRB.CreateAlloca(RetTy);
      IRB.CreateAlignedStore(CB, RetSpill, DL.getStackAlignment());

      // Add the spilled return value
      AfterHookParamTys.push_back(RetSpill->getType());
      AfterHookParamVals.push_back(RetSpill);
    }
  }
  for (Value *Arg : CB->args()) {
    Type *ArgTy = Arg->getType();
    if (!needToSpillType(ArgTy)) {
      // We can simply pass the argument directly to the hook.
      AfterHookParamTys.push_back(ArgTy);
      AfterHookParamVals.push_back(Arg);
      continue;
    }
    // We need to spill the argument onto the stack.

    // Save the stack pointer, if we haven't already
    if (!SavedStack)
      SavedStack = IRB.CreateStackSave();

    // Spill the argument onto the stack
    AllocaInst *ArgSpill = IRB.CreateAlloca(ArgTy);
    IRB.CreateAlignedStore(Arg, ArgSpill, DL.getStackAlignment());

    // Add the spilled argument
    AfterHookParamTys.push_back(ArgSpill->getType());
    AfterHookParamVals.push_back(ArgSpill);
  }

  // Special-case intrinsics.
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
  switch (II->getIntrinsicID()) {
  case Intrinsic::hyper_lookup_0:
  case Intrinsic::hyper_lookup_1:
  case Intrinsic::hyper_lookup_2: {
    FunctionType *AfterHookTy =
        FunctionType::get(IRB.getPtrTy(), AfterHookParamTys, Called->isVarArg());
    FunctionCallee AfterIntrinCallHook = getOrInsertSynthesizedHook(
        ("__csan_" + Buf).str(), AfterHookTy, ReturnParam);
    CallInst *HookCall =
        insertHookCall(&*Iter, AfterIntrinCallHook, AfterHookParamVals);
    II->replaceUsesWithIf(HookCall, [HookCall](Use &U) {
      return cast<Instruction>(U.getUser()) != HookCall;
    });
    return true;
  }
  }

  FunctionType *AfterHookTy =
      FunctionType::get(IRB.getVoidTy(), AfterHookParamTys, Called->isVarArg());
  FunctionCallee AfterIntrinCallHook = getOrInsertSynthesizedHook(
      ("__csan_" + Buf).str(), AfterHookTy, ReturnParam);

  // Insert the hook call
  insertHookCall(&*Iter, AfterIntrinCallHook, AfterHookParamVals);

  if (SavedStack)
    IRB.CreateStackRestore(SavedStack);

  return true;
}

bool CilkSanitizerImpl::instrumentLibCall(Instruction *I,
                                          SmallVectorImpl<Value *> *MAAPVals) {
  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SERIESPARALLEL))
    return true;

  bool IsInvoke = isa<InvokeInst>(I);
  CallBase *CB = dyn_cast<CallBase>(I);
  if (!CB)
    return false;
  Function *Called = CB->getCalledFunction();

  IRBuilder<> IRB(I);
  LLVMContext &Ctx = IRB.getContext();
  uint64_t LocalId = CallsiteFED.add(*I);
  Value *DefaultID = getDefaultID(IRB);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = getCalleeFuncID(Called, IRB);
  assert(FuncId != NULL);

  Value *NumMVVal = IRB.getInt8(0);
  if (MAAPVals && !MAAPVals->empty()) {
    unsigned NumMV = MAAPVals->size();
    NumMVVal = IRB.getInt8(NumMV);
  }

  CsiCallProperty Prop;
  Value *DefaultPropVal = Prop.getValue(IRB);
  // TODO: Set appropriate property values for this intrinsic call
  Value *PropVal = Prop.getValue(IRB);
  Type *IDType = IRB.getInt64Ty();

  // If the intrinsic does not return, insert the hook before the intrinsic.
  if (CB->doesNotReturn()) {
    // Synthesize the before hook for this function.
    SmallVector<Type *, 8> BeforeHookParamTys(
        {IDType, /*callee func_id*/ IDType,
         /*MAAP_count*/ IRB.getInt8Ty(), CsiCallProperty::getType(Ctx)});
    SmallVector<Value *, 8> BeforeHookParamVals(
        {CallsiteId, FuncId, NumMVVal, PropVal});
    BeforeHookParamTys.append(Called->getFunctionType()->param_begin(),
                              Called->getFunctionType()->param_end());
    BeforeHookParamVals.append(CB->arg_begin(), CB->arg_end());
    FunctionType *BeforeHookTy = FunctionType::get(
        IRB.getVoidTy(), BeforeHookParamTys, Called->isVarArg());
    FunctionCallee BeforeLibCallHook = getOrInsertSynthesizedHook(
        ("__csan_" + Called->getName()).str(), BeforeHookTy);

    insertHookCall(I, BeforeLibCallHook, BeforeHookParamVals);
    return true;
  }

  // Otherwise, insert the hook after the intrinsic.

  // Synthesize the after hook for this function.
  SmallVector<Type *, 8> AfterHookParamTys(
      {IDType, /*callee func_id*/ IDType,
       /*Num MAAPVal*/ IRB.getInt8Ty(), CsiCallProperty::getType(Ctx)});
  SmallVector<Value *, 8> AfterHookParamVals(
      {CallsiteId, FuncId, NumMVVal, PropVal});
  SmallVector<Value *, 8> AfterHookDefaultVals(
      {DefaultID, DefaultID, IRB.getInt8(0), DefaultPropVal});
  int ReturnParam = -1;
  if (!Called->getReturnType()->isVoidTy()) {
    ReturnParam = AfterHookParamTys.size();
    AfterHookParamTys.push_back(Called->getReturnType());
    AfterHookParamVals.push_back(CB);
    AfterHookDefaultVals.push_back(
        Constant::getNullValue(Called->getReturnType()));
  }
  AfterHookParamTys.append(Called->getFunctionType()->param_begin(),
                           Called->getFunctionType()->param_end());
  AfterHookParamVals.append(CB->arg_begin(), CB->arg_end());
  for (Value *Arg : CB->args())
    AfterHookDefaultVals.push_back(Constant::getNullValue(Arg->getType()));
  FunctionType *AfterHookTy =
      FunctionType::get(IRB.getVoidTy(), AfterHookParamTys, Called->isVarArg());
  FunctionCallee AfterLibCallHook = getOrInsertSynthesizedHook(
      ("__csan_" + Called->getName()).str(), AfterHookTy, ReturnParam);

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);
    insertHookCallInSuccessorBB(
        II->getNormalDest(), II->getParent(), AfterLibCallHook,
        AfterHookParamVals, AfterHookDefaultVals);
    // Don't insert any instrumentation in the exception block.
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    insertHookCall(&*Iter, AfterLibCallHook, AfterHookParamVals);
  }

  return true;
}

bool CilkSanitizerImpl::instrumentCallsite(Instruction *I,
                                           SmallVectorImpl<Value *> *MAAPVals) {
  if (callsPlaceholderFunction(*I))
    return false;

  bool IsInvoke = isa<InvokeInst>(I);
  CallBase *CB = dyn_cast<CallBase>(I);
  if (!CB)
    return false;
  Function *Called = CB->getCalledFunction();

  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SERIESPARALLEL))
    return true;

  IRBuilder<> IRB(I);
  uint64_t LocalId = CallsiteFED.add(*I);
  Value *DefaultID = getDefaultID(IRB);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = getCalleeFuncID(Called, IRB);
  assert(FuncId != NULL);

  Value *NumMVVal = IRB.getInt8(0);
  if (MAAPVals && !MAAPVals->empty()) {
    unsigned NumMV = MAAPVals->size();
    NumMVVal = IRB.getInt8(NumMV);
  }

  CsiCallProperty Prop;
  Value *DefaultPropVal = Prop.getValue(IRB);
  Prop.setIsIndirect(!Called);
  Value *PropVal = Prop.getValue(IRB);
  insertHookCall(I, CsanBeforeCallsite, {CallsiteId, FuncId, NumMVVal,
                                         PropVal});

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);
    if (!CB->doesNotReturn()) {
      // If this function can return normally, insert an after_call hook at the
      // normal destination.
      insertHookCallInSuccessorBB(
          II->getNormalDest(), II->getParent(), CsanAfterCallsite,
          {CallsiteId, FuncId, NumMVVal, PropVal},
          {DefaultID, DefaultID, IRB.getInt8(0), DefaultPropVal});
    }
    CsiCallProperty Prop;
    Prop.setIsIndirect(!Called);
    Prop.setIsUnwind();
    Value *PropVal = Prop.getValue(IRB);
    insertHookCallInSuccessorBB(
        II->getUnwindDest(), II->getParent(), CsanAfterCallsite,
        {CallsiteId, FuncId, NumMVVal, PropVal},
        {DefaultID, DefaultID, IRB.getInt8(0), DefaultPropVal});
  } else if (!CB->doesNotReturn()) {
    // If this function can return normally, insert an after_call hook at the
    // normal destination.

    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    PropVal = Prop.getValue(IRB);
    insertHookCall(&*Iter, CsanAfterCallsite,
                   {CallsiteId, FuncId, NumMVVal, PropVal});
  }

  return true;
}

bool CilkSanitizerImpl::suppressCallsite(Instruction *I) {
  if (callsPlaceholderFunction(*I))
    return false;

  bool IsInvoke = isa<InvokeInst>(I);

  IRBuilder<> IRB(I);
  insertHookCall(I, CsanDisableChecking, {});

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);
    insertHookCallInSuccessorBB(
        II->getNormalDest(), II->getParent(), CsanEnableChecking, {}, {});
    insertHookCallInSuccessorBB(
        II->getUnwindDest(), II->getParent(), CsanEnableChecking, {}, {});
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    insertHookCall(&*Iter, CsanEnableChecking, {});
  }

  return true;
}

static bool isMemTransferDstOperand(unsigned OperandNum) {
  // This check should be kept in sync with TapirRaceDetect::GetGeneralAccesses.
  return (OperandNum == 0);
}

static bool isMemTransferSrcOperand(unsigned OperandNum) {
  // This check should be kept in sync with TapirRaceDetect::GetGeneralAccesses.
  return (OperandNum == 1);
}

bool CilkSanitizerImpl::instrumentAnyMemIntrinAcc(Instruction *I,
                                                  unsigned OperandNum,
                                                  IRBuilder<> &IRB) {
  CsiLoadStoreProperty Prop;
  if (AnyMemTransferInst *M = dyn_cast<AnyMemTransferInst>(I)) {
    // Only instrument the large load and the large store components as
    // necessary.
    bool Instrumented = false;

    if (isMemTransferDstOperand(OperandNum)) {
      // Only insert instrumentation if requested
      if (!(InstrumentationSet & SHADOWMEMORY))
        return true;

      Value *Addr = M->getDest();
      Prop.setAlignment(M->getDestAlign());
      Prop.setIsThreadLocal(isThreadLocalObject(lookupUnderlyingObject(Addr)));
      // Instrument the store
      uint64_t StoreId = StoreFED.add(*I);

      // TODO: Don't recalculate underlying objects
      uint64_t StoreObjId = StoreObj.add(*I, lookupUnderlyingObject(Addr));
      assert(StoreId == StoreObjId &&
             "Store received different ID's in FED and object tables.");

      Value *CsiId = StoreFED.localToGlobalId(StoreId, IRB);
      SmallVector<Value *, 5> Args = {
          CsiId, IRB.CreatePointerCast(Addr, IRB.getPtrTy()),
          IRB.CreateIntCast(M->getLength(), IntptrTy, false),
          Prop.getValue(IRB)};
      if (LargeStoreTakesLabel) {
        CallInst *CurLabCall = getCurOSLabelCall(IRB);
        Args.push_back(CurLabCall);
      }
      Instruction *Call = IRB.CreateCall(CsanLargeWrite, Args);
      IRB.SetInstDebugLocation(Call);
      ++NumInstrumentedMemIntrinsicWrites;
      Instrumented = true;
    }

    if (isMemTransferSrcOperand(OperandNum)) {
      // Only insert instrumentation if requested
      if (!(InstrumentationSet & SHADOWMEMORY))
        return true;

      Value *Addr = M->getSource();
      Prop.setAlignment(M->getSourceAlign());
      Prop.setIsThreadLocal(isThreadLocalObject(lookupUnderlyingObject(Addr)));
      // Instrument the load
      uint64_t LoadId = LoadFED.add(*I);

      // TODO: Don't recalculate underlying objects
      uint64_t LoadObjId = LoadObj.add(*I, lookupUnderlyingObject(Addr));
      assert(LoadId == LoadObjId &&
             "Load received different ID's in FED and object tables.");

      Value *CsiId = LoadFED.localToGlobalId(LoadId, IRB);
      SmallVector<Value *, 5> Args = {
          CsiId, IRB.CreatePointerCast(Addr, IRB.getPtrTy()),
          IRB.CreateIntCast(M->getLength(), IntptrTy, false),
          Prop.getValue(IRB)};
      if (LargeLoadTakesLabel) {
        CallInst *CurLabCall = getCurOSLabelCall(IRB);
        Args.push_back(CurLabCall);
      }
      Instruction *Call = IRB.CreateCall(CsanLargeRead, Args);
      IRB.SetInstDebugLocation(Call);
      ++NumInstrumentedMemIntrinsicReads;
      Instrumented = true;
    }
    return Instrumented;
  }
  if (AnyMemIntrinsic *M = dyn_cast<AnyMemIntrinsic>(I)) {
    // Only insert instrumentation if requested
    if (!(InstrumentationSet & SHADOWMEMORY))
      return true;

    Value *Addr = M->getDest();
    Prop.setAlignment(M->getDestAlign());
    Prop.setIsThreadLocal(isThreadLocalObject(lookupUnderlyingObject(Addr)));
    uint64_t LocalId = StoreFED.add(*I);

    // TODO: Don't recalculate underlying objects
    uint64_t StoreObjId = StoreObj.add(*I, lookupUnderlyingObject(Addr));
    assert(LocalId == StoreObjId &&
           "Store received different ID's in FED and object tables.");

    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    SmallVector<Value *, 5> Args = {
        CsiId, IRB.CreatePointerCast(Addr, IRB.getPtrTy()),
        IRB.CreateIntCast(M->getLength(), IntptrTy, false),
        Prop.getValue(IRB)};
    if (LargeStoreTakesLabel) {
      CallInst *CurLabCall = getCurOSLabelCall(IRB);
      Args.push_back(CurLabCall);
    }
    Instruction *Call = IRB.CreateCall(CsanLargeWrite, Args);
    IRB.SetInstDebugLocation(Call);
    ++NumInstrumentedMemIntrinsicWrites;
    return true;
  }
  return false;
}

static void getTaskExits(
    DetachInst *DI, SmallVectorImpl<BasicBlock *> &TaskReturns,
    SmallVectorImpl<BasicBlock *> &TaskResumes,
    SmallVectorImpl<Spindle *> &SharedEHExits,
    TaskInfo &TI) {
  BasicBlock *DetachedBlock = DI->getDetached();
  Task *T = TI.getTaskFor(DetachedBlock);
  BasicBlock *ContinueBlock = DI->getContinue();

  // Examine the predecessors of the continue block and save any predecessors in
  // the task as a task return.
  for (BasicBlock *Pred : predecessors(ContinueBlock)) {
    if (T->simplyEncloses(Pred)) {
      assert(isa<ReattachInst>(Pred->getTerminator()));
      TaskReturns.push_back(Pred);
    }
  }

  // If the detach cannot throw, we're done.
  if (!DI->hasUnwindDest())
    return;

  // Detached-rethrow exits can appear in strange places within a task-exiting
  // spindle.  Hence we loop over all blocks in the spindle to find
  // detached rethrows.
  for (Spindle *S : depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
    if (S->isSharedEH()) {
      if (llvm::any_of(predecessors(S),
                       [](const Spindle *Pred){ return !Pred->isSharedEH(); }))
        SharedEHExits.push_back(S);
      continue;
    }

    for (BasicBlock *B : S->blocks())
      if (isDetachedRethrow(B->getTerminator()))
        TaskResumes.push_back(B);
  }
}

bool CilkSanitizerImpl::instrumentDetach(DetachInst *DI, unsigned SyncRegNum,
                                         unsigned NumSyncRegs,
                                         DominatorTree &DT, TaskInfo &TI,
                                         LoopInfo &LI) {
  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SERIESPARALLEL))
    return true;

  LLVMContext &Ctx = DI->getContext();
  BasicBlock *TaskEntryBlock = TI.getTaskFor(DI->getParent())->getEntry();
  IRBuilder<> IDBuilder(getEntryBBInsertPt(*TaskEntryBlock));
  bool TapirLoopBody = spawnsTapirLoopBody(DI, LI, TI);
  ConstantInt *SyncRegVal = ConstantInt::get(Type::getInt32Ty(Ctx), SyncRegNum);
  ConstantInt *DefaultSyncRegVal = ConstantInt::get(Type::getInt32Ty(Ctx), 0);
  CsiDetachProperty DetachProp;
  DetachProp.setForTapirLoopBody(TapirLoopBody);
  // Instrument the detach instruction itself
  Value *DetachID;
  {
    IRBuilder<> IRB(DI);
    uint64_t LocalID = DetachFED.add(*DI);
    DetachID = DetachFED.localToGlobalId(LocalID, IDBuilder);
    Instruction *Call = IRB.CreateCall(
        CsanDetach, {DetachID, SyncRegVal, DetachProp.getValue(IRB)});
    IRB.SetInstDebugLocation(Call);
  }
  NumInstrumentedDetaches++;

  // Find the detached block, continuation, and associated reattaches.
  BasicBlock *DetachedBlock = DI->getDetached();
  BasicBlock *ContinueBlock = DI->getContinue();
  Task *T = TI.getTaskFor(DetachedBlock);
  SmallVector<BasicBlock *, 8> TaskExits, TaskResumes;
  SmallVector<Spindle *, 2> SharedEHExits;
  getTaskExits(DI, TaskExits, TaskResumes, SharedEHExits, TI);

  // Instrument the entry and exit points of the detached task.
  {
    // Instrument the entry point of the detached task.
    IRBuilder<> IRB(&*getFirstInsertionPtInDetachedBlock(DetachedBlock));
    if (!IRB.getCurrentDebugLocation())
      IRB.SetCurrentDebugLocation(searchForDebugLoc(&*IRB.GetInsertPoint()));
    uint64_t LocalID = TaskFED.add(*DetachedBlock);
    Value *TaskID = TaskFED.localToGlobalId(LocalID, IDBuilder);
    CsiTaskProperty Prop;
    Prop.setIsTapirLoopBody(TapirLoopBody);
    Prop.setNumSyncReg(NumSyncRegs);
    // Get the frame and stack pointers.
    Value *FrameAddr = IRB.CreateCall(
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::task_frameaddress),
        {IRB.getInt32(0)});
    Value *StackSave = IRB.CreateStackSave();
    Instruction *Call = IRB.CreateCall(CsanTaskEntry,
                                       {TaskID, DetachID, FrameAddr,
                                        StackSave, Prop.getValue(IRB)});
    IRB.SetInstDebugLocation(Call);

    // Instrument the exit points of the detached tasks.
    for (BasicBlock *TaskExit : TaskExits) {
      IRBuilder<> IRB(TaskExit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*TaskExit->getTerminator());
      Value *TaskExitID = TaskExitFED.localToGlobalId(LocalID, IDBuilder);
      CsiTaskExitProperty ExitProp;
      ExitProp.setIsTapirLoopBody(TapirLoopBody);
      Instruction *Call =
          IRB.CreateCall(CsanTaskExit, {TaskExitID, TaskID, DetachID,
                                        SyncRegVal, ExitProp.getValue(IRB)});
      IRB.SetInstDebugLocation(Call);
      NumInstrumentedDetachExits++;
    }
    // Instrument the EH exits of the detached task.
    for (BasicBlock *TaskExit : TaskResumes) {
      IRBuilder<> IRB(TaskExit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*TaskExit->getTerminator());
      Value *TaskExitID = TaskExitFED.localToGlobalId(LocalID, IDBuilder);
      CsiTaskExitProperty ExitProp;
      ExitProp.setIsTapirLoopBody(TapirLoopBody);
      Instruction *Call =
          IRB.CreateCall(CsanTaskExit, {TaskExitID, TaskID, DetachID,
                                        SyncRegVal, ExitProp.getValue(IRB)});
      IRB.SetInstDebugLocation(Call);
      NumInstrumentedDetachExits++;
    }

    Value *DefaultID = getDefaultID(IDBuilder);
    for (Spindle *SharedEH : SharedEHExits) {
      // Skip shared-eh spindle exits that are placeholder unreachable blocks.
      if (isa<UnreachableInst>(
              SharedEH->getEntry()->getFirstNonPHIOrDbgOrLifetime()))
        continue;
      CsiTaskExitProperty ExitProp;
      ExitProp.setIsTapirLoopBody(TapirLoopBody);
      insertHookCallAtSharedEHSpindleExits(
          SharedEH, T, CsanTaskExit, TaskExitFED,
          {TaskID, DetachID, SyncRegVal, ExitProp.getValueImpl(Ctx)},
          {DefaultID, DefaultID, DefaultSyncRegVal,
           CsiTaskExitProperty::getDefaultValueImpl(Ctx)});
    }
  }

  // Instrument the continuation of the detach.
  {
    if (isCriticalContinueEdge(DI, 1))
      ContinueBlock = SplitCriticalEdge(
          DI, 1,
          CriticalEdgeSplittingOptions(&DT, &LI).setSplitDetachContinue());

    IRBuilder<> IRB(&*ContinueBlock->getFirstInsertionPt());
    if (!IRB.getCurrentDebugLocation())
      IRB.SetCurrentDebugLocation(searchForDebugLoc(&*IRB.GetInsertPoint()));
    uint64_t LocalID = DetachContinueFED.add(*ContinueBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IDBuilder);
    CsiDetachContinueProperty ContProp;
    ContProp.setForTapirLoopBody(TapirLoopBody);
    Instruction *Call =
        IRB.CreateCall(CsanDetachContinue, {ContinueID, DetachID, SyncRegVal,
                                            ContProp.getValue(IRB)});
    IRB.SetInstDebugLocation(Call);
  }
  // Instrument the unwind of the detach, if it exists.
  if (DI->hasUnwindDest()) {
    BasicBlock *UnwindBlock = DI->getUnwindDest();
    BasicBlock *PredBlock = DI->getParent();
    if (Value *TF = T->getTaskFrameUsed()) {
      // If the detached task uses a taskframe, then we want to insert the
      // detach_continue instrumentation for the unwind destination after the
      // taskframe.resume.
      UnwindBlock = getTaskFrameResumeDest(TF);
      assert(UnwindBlock &&
             "Detach with unwind uses a taskframe with no resume");
      PredBlock = getTaskFrameResume(TF)->getParent();
    }
    Value *DefaultID = getDefaultID(IDBuilder);
    uint64_t LocalID = DetachContinueFED.add(*UnwindBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IDBuilder);
    CsiDetachContinueProperty ContProp;
    Value *DefaultPropVal = ContProp.getValueImpl(Ctx);
    ContProp.setIsUnwind();
    ContProp.setForTapirLoopBody(TapirLoopBody);
    insertHookCallInSuccessorBB(
        UnwindBlock, PredBlock, CsanDetachContinue,
        {ContinueID, DetachID, SyncRegVal, ContProp.getValue(Ctx)},
        {DefaultID, DefaultID, DefaultSyncRegVal, DefaultPropVal});
    for (BasicBlock *DRPred : predecessors(UnwindBlock))
      if (isDetachedRethrow(DRPred->getTerminator(), DI->getSyncRegion()))
        insertHookCallInSuccessorBB(
            UnwindBlock, DRPred, CsanDetachContinue,
            {ContinueID, DetachID, SyncRegVal, ContProp.getValue(Ctx)},
            {DefaultID, DefaultID, DefaultSyncRegVal, DefaultPropVal});
  }
  return true;
}

bool CilkSanitizerImpl::instrumentSync(SyncInst *SI, unsigned SyncRegNum) {
  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SERIESPARALLEL))
    return true;

  IRBuilder<> IRB(SI);
  Value *DefaultID = getDefaultID(IRB);
  // Get the ID of this sync.
  uint64_t LocalID = SyncFED.add(*SI);
  Value *SyncID = SyncFED.localToGlobalId(LocalID, IRB);
  // Insert instrumentation before the sync.
  insertHookCall(SI, CsanBeforeSync, {SyncID, IRB.getInt32(SyncRegNum)});

  // NOTE: Because Cilksan executes serially, any exceptions thrown before this
  // sync will appear to be thrown from their respective spawns or calls, not
  // the sync or the Cilk personality function.  Hence we don't need
  // instrumentation in the unwind destination of the sync.

  NumInstrumentedSyncs++;
  return true;
}

void CilkSanitizerImpl::instrumentTapirLoop(Loop &L, TaskInfo &TI,
                                       DenseMap<Value *, unsigned> &SyncRegNums,
                                       ScalarEvolution *SE) {
  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SERIESPARALLEL))
    return;

  assert(L.isLoopSimplifyForm() && "CSI assumes loops are in simplified form.");
  BasicBlock *Preheader = L.getLoopPreheader();
  Task *T = getTaskIfTapirLoop(&L, &TI);
  assert(T && "CilkSanitizer should only instrument Tapir loops.");
  unsigned SyncRegNum = SyncRegNums[T->getDetach()->getSyncRegion()];
  // We assign a local ID for this loop here, so that IDs for loops follow a
  // depth-first ordering.
  csi_id_t LocalId = LoopFED.add(*T->getDetach());

  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L.getExitingBlocks(ExitingBlocks);
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L.getUniqueExitBlocks(ExitBlocks);

  // Record properties of this loop.
  CsiLoopProperty LoopProp;
  LoopProp.setIsTapirLoop(static_cast<bool>(getTaskIfTapirLoop(&L, &TI)));
  LoopProp.setHasUniqueExitingBlock((ExitingBlocks.size() == 1));

  IRBuilder<> IRB(Preheader->getTerminator());
  Value *LoopCsiId = LoopFED.localToGlobalId(LocalId, IRB);
  Value *LoopPropVal = LoopProp.getValue(IRB);

  // Try to evaluate the runtime trip count for this loop.  Default to a count
  // of -1 for unknown trip counts.
  Value *TripCount = IRB.getInt64(-1);
  if (SE) {
    const SCEV *TripCountSC = getRuntimeTripCount(L, SE, true);
    if (!isa<SCEVCouldNotCompute>(TripCountSC)) {
      // Extend the TripCount type if necessary.
      if (TripCountSC->getType() != IRB.getInt64Ty())
        TripCountSC = SE->getZeroExtendExpr(TripCountSC, IRB.getInt64Ty());
      // Compute the trip count to pass to the CSI hook.
      SCEVExpander Expander(*SE, DL, "csi");
      TripCount = Expander.expandCodeFor(TripCountSC, IRB.getInt64Ty(),
                                         &*IRB.GetInsertPoint());
    }
  }

  // Insert before-loop hook.
  insertHookCall(&*IRB.GetInsertPoint(), CsanBeforeLoop, {LoopCsiId, TripCount,
                                                          LoopPropVal});

  // Insert after-loop hooks.
  for (BasicBlock *BB : ExitBlocks) {
    // If the exit block is simply enclosed inside the task, then its on an
    // exceptional exit path from the task.  In that case, the exit path will
    // reach the unwind destination of the detach.  Because the unwind
    // destination of the detach is in the set of exit blocks, we can safely
    // skip any exit blocks enclosed in the task.
    if (!T->encloses(BB)) {
      IRB.SetInsertPoint(&*BB->getFirstInsertionPt());
      insertHookCall(&*IRB.GetInsertPoint(), CsanAfterLoop,
                     {LoopCsiId, IRB.getInt8(SyncRegNum), LoopPropVal});
    }
  }
}

bool CilkSanitizerImpl::instrumentAlloca(Instruction *I, TaskInfo &TI) {
  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SHADOWMEMORY))
    return true;

  IRBuilder<> IRB(I);
  bool InsertingAtAlloca = true;
  if (isEntryBlock(*I->getParent(), TI)) {
    Instruction *EntryBBInsertPt = getEntryBBInsertPt(*I->getParent());
    if (I->comesBefore(EntryBBInsertPt)) {
      IRB.SetInsertPoint(EntryBBInsertPt);
      InsertingAtAlloca = false;
    }
  }
  AllocaInst *AI = cast<AllocaInst>(I);

  uint64_t LocalId = AllocaFED.add(*I);
  Value *CsiId = AllocaFED.localToGlobalId(LocalId, IRB);
  uint64_t AllocaObjId = AllocaObj.add(*I, I);
  assert(LocalId == AllocaObjId &&
         "Alloca received different ID's in FED and object tables.");

  CsiAllocaProperty Prop;
  Prop.setIsStatic(AI->isStaticAlloca());
  Value *PropVal = Prop.getValue(IRB);

  // Get size of allocation.
  uint64_t Size = DL.getTypeAllocSize(AI->getAllocatedType());
  Value *SizeVal = IRB.getInt64(Size);
  if (AI->isArrayAllocation())
    SizeVal = IRB.CreateMul(SizeVal,
                            IRB.CreateZExtOrBitCast(AI->getArraySize(),
                                                    IRB.getInt64Ty()));

  BasicBlock::iterator Iter(I);
  if (InsertingAtAlloca) {
    Iter++;
    IRB.SetInsertPoint(&*Iter);
  } else {
    Iter = IRB.GetInsertPoint();
  }

  Type *AddrType = IRB.getPtrTy();
  Value *Addr = IRB.CreatePointerCast(I, AddrType);
  insertHookCall(&*Iter, CsiAfterAlloca, {CsiId, Addr, SizeVal, PropVal});

  NumInstrumentedAllocas++;
  return true;
}

static Value *getHeapObject(Value *I) {
  Value *Object = nullptr;
  unsigned NumOfBitCastUses = 0;

  // Determine if CallInst has a bitcast use.
  for (Value::user_iterator UI = I->user_begin(), E = I->user_end();
       UI != E;)
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(*UI++)) {
      // Look for a dbg.value intrinsic for this bitcast.
      SmallVector<DbgValueInst *, 1> DbgValues;
      findDbgValues(DbgValues, BCI);
      if (!DbgValues.empty()) {
        Object = BCI;
        NumOfBitCastUses++;
      }
    }

  // Heap-allocation call has 1 debug-bitcast use, so use that bitcast as the
  // object.
  if (NumOfBitCastUses == 1)
    return Object;

  // Otherwise just use the heap-allocation call directly.
  return I;
}

bool CilkSanitizerImpl::instrumentAllocFnLibCall(Instruction *I,
                                                 const TargetLibraryInfo *TLI) {
  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SHADOWMEMORY))
    return true;

  bool IsInvoke = isa<InvokeInst>(I);
  CallBase *CB = dyn_cast<CallBase>(I);
  if (!CB)
    return false;
  Function *Called = CB->getCalledFunction();

  // Get the CSI IDs for this hook
  IRBuilder<> IRB(I);
  LLVMContext &Ctx = IRB.getContext();
  Value *DefaultID = getDefaultID(IRB);
  uint64_t LocalId = AllocFnFED.add(*I);
  Value *AllocFnId = AllocFnFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = getCalleeFuncID(Called, IRB);
  assert(FuncId != NULL);

  // Get the ID for the corresponding heap object
  Value *HeapObj = nullptr;
  if ("posix_memalign" == Called->getName())
    HeapObj = getHeapObject(CB->getArgOperand(0));
  else
    HeapObj = getHeapObject(I);
  uint64_t AllocFnObjId = AllocFnObj.add(*I, HeapObj);
  assert(LocalId == AllocFnObjId &&
         "Allocation fn received different ID's in FED and object tables.");

  // TODO: Propagate MAAPs to allocation-function library calls
  Value *NumMVVal = IRB.getInt8(0);

  CsiAllocFnProperty Prop;
  Value *DefaultPropVal = Prop.getValue(IRB);
  LibFunc AllocLibF;
  TLI->getLibFunc(*Called, AllocLibF);
  Prop.setAllocFnTy(static_cast<unsigned>(getAllocFnTy(AllocLibF)));
  Value *PropVal = Prop.getValue(IRB);
  Type *IDType = IRB.getInt64Ty();

  // Synthesize the after hook for this function.
  SmallVector<Type *, 8> AfterHookParamTys({IDType, /*callee func_id*/ IDType,
                                            /*MAAP_count*/ IRB.getInt8Ty(),
                                            CsiAllocFnProperty::getType(Ctx)});
  SmallVector<Value *, 8> AfterHookParamVals(
      {AllocFnId, FuncId, NumMVVal, PropVal});
  SmallVector<Value *, 8> AfterHookDefaultVals(
      {DefaultID, DefaultID, IRB.getInt8(0), DefaultPropVal});
  int ReturnParam = -1;
  if (!Called->getReturnType()->isVoidTy()) {
    ReturnParam = AfterHookParamVals.size();
    AfterHookParamTys.push_back(Called->getReturnType());
    AfterHookParamVals.push_back(CB);
    AfterHookDefaultVals.push_back(
        Constant::getNullValue(Called->getReturnType()));
  }
  AfterHookParamTys.append(Called->getFunctionType()->param_begin(),
                           Called->getFunctionType()->param_end());
  AfterHookParamVals.append(CB->arg_begin(), CB->arg_end());
  for (Value *Arg : CB->args())
    AfterHookDefaultVals.push_back(Constant::getNullValue(Arg->getType()));
  FunctionType *AfterHookTy =
      FunctionType::get(IRB.getVoidTy(), AfterHookParamTys, Called->isVarArg());
  FunctionCallee AfterLibCallHook = getOrInsertSynthesizedHook(
      ("__csan_alloc_" + Called->getName()).str(), AfterHookTy, ReturnParam);

  // Insert the hook after the call.
  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);
    insertHookCallInSuccessorBB(
        II->getNormalDest(), II->getParent(), AfterLibCallHook,
        AfterHookParamVals, AfterHookDefaultVals);
    // Don't insert any instrumentation in the exception block.
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    insertHookCall(&*Iter, AfterLibCallHook, AfterHookParamVals);
  }

  NumInstrumentedAllocFns++;
  return true;
}

bool CilkSanitizerImpl::instrumentAllocationFn(Instruction *I,
                                               DominatorTree &DT,
                                               const TargetLibraryInfo *TLI) {
  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SHADOWMEMORY))
    return true;

  bool IsInvoke = isa<InvokeInst>(I);
  assert(isa<CallBase>(I) &&
         "instrumentAllocationFn not given a call or invoke instruction.");
  Function *Called = dyn_cast<CallBase>(I)->getCalledFunction();
  assert(Called && "Could not get called function for allocation fn.");

  IRBuilder<> IRB(I);
  SmallVector<Value *, 4> AllocFnArgs;
  if (!getAllocFnArgs(I, AllocFnArgs, IntptrTy, IRB.getPtrTy(), *TLI)) {
    return instrumentAllocFnLibCall(I, TLI);
  }
  SmallVector<Value *, 4> DefaultAllocFnArgs(
      {/* Allocated size */ Constant::getNullValue(IntptrTy),
       /* Number of elements */ Constant::getNullValue(IntptrTy),
       /* Alignment */ Constant::getNullValue(IntptrTy),
       /* Old pointer */ Constant::getNullValue(IRB.getPtrTy()),});

  Value *DefaultID = getDefaultID(IRB);
  uint64_t LocalId = AllocFnFED.add(*I);
  Value *AllocFnId = AllocFnFED.localToGlobalId(LocalId, IRB);
  uint64_t AllocFnObjId = AllocFnObj.add(*I, getHeapObject(I));
  assert(LocalId == AllocFnObjId &&
         "Allocation fn received different ID's in FED and object tables.");

  CsiAllocFnProperty Prop;
  Value *DefaultPropVal = Prop.getValue(IRB);
  LibFunc AllocLibF;
  TLI->getLibFunc(*Called, AllocLibF);
  Prop.setAllocFnTy(static_cast<unsigned>(getAllocFnTy(AllocLibF)));
  AllocFnArgs.push_back(Prop.getValue(IRB));
  DefaultAllocFnArgs.push_back(DefaultPropVal);

  // Only calls with an old pointer (realloc) get the before hook.
  if (!isa<ConstantPointerNull>(AllocFnArgs[3])) {
    SmallVector<Value *, 6> BeforeAllocFnArgs({AllocFnId});
    BeforeAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
    insertHookCall(I, CsanBeforeAllocFn, BeforeAllocFnArgs);
  }

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);

    BasicBlock *NormalBB = II->getNormalDest();
    unsigned SuccNum = GetSuccessorNumber(II->getParent(), NormalBB);
    if (isCriticalEdge(II, SuccNum))
      NormalBB = SplitCriticalEdge(II, SuccNum,
                                   CriticalEdgeSplittingOptions(&DT));
    // Insert hook into normal destination.
    {
      IRB.SetInsertPoint(&*NormalBB->getFirstInsertionPt());
      SmallVector<Value *, 4> AfterAllocFnArgs;
      AfterAllocFnArgs.push_back(AllocFnId);
      AfterAllocFnArgs.push_back(IRB.CreatePointerCast(I, IRB.getPtrTy()));
      AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
      insertHookCall(&*IRB.GetInsertPoint(), CsanAfterAllocFn,
                     AfterAllocFnArgs);
    }
    // Insert hook into unwind destination.
    {
      // The return value of the allocation function is not valid in the unwind
      // destination.
      SmallVector<Value *, 4> AfterAllocFnArgs, DefaultAfterAllocFnArgs;
      AfterAllocFnArgs.push_back(AllocFnId);
      AfterAllocFnArgs.push_back(Constant::getNullValue(IRB.getPtrTy()));
      AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
      DefaultAfterAllocFnArgs.push_back(DefaultID);
      DefaultAfterAllocFnArgs.push_back(
          Constant::getNullValue(IRB.getPtrTy()));
      DefaultAfterAllocFnArgs.append(DefaultAllocFnArgs.begin(),
                                     DefaultAllocFnArgs.end());
      insertHookCallInSuccessorBB(
          II->getUnwindDest(), II->getParent(), CsanAfterAllocFn,
          AfterAllocFnArgs, DefaultAfterAllocFnArgs);
    }
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    SmallVector<Value *, 4> AfterAllocFnArgs;
    AfterAllocFnArgs.push_back(AllocFnId);
    AfterAllocFnArgs.push_back(IRB.CreatePointerCast(I, IRB.getPtrTy()));
    AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
    insertHookCall(&*Iter, CsanAfterAllocFn, AfterAllocFnArgs);
  }

  NumInstrumentedAllocFns++;
  return true;
}

bool CilkSanitizerImpl::instrumentFree(Instruction *I,
                                       const TargetLibraryInfo *TLI) {
  // Only insert instrumentation if requested
  if (!(InstrumentationSet & SHADOWMEMORY))
    return true;

  // It appears that frees (and deletes) never throw.
  assert(isa<CallInst>(I) && "Free call is not a call instruction");

  CallInst *FC = cast<CallInst>(I);
  Function *Called = FC->getCalledFunction();
  assert(Called && "Could not get called function for free.");

  IRBuilder<> IRB(I);
  uint64_t LocalId = FreeFED.add(*I);
  Value *FreeId = FreeFED.localToGlobalId(LocalId, IRB);

  // All currently supported free functions free the first argument.
  Value *Addr = FC->getArgOperand(0);
  CsiFreeProperty Prop;
  LibFunc FreeLibF;
  TLI->getLibFunc(*Called, FreeLibF);
  Prop.setFreeTy(static_cast<unsigned>(getFreeTy(FreeLibF)));

  insertHookCall(I, CsanBeforeFree, {FreeId, Addr, Prop.getValue(IRB)});

  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);
  insertHookCall(&*Iter, CsanAfterFree, {FreeId, Addr, Prop.getValue(IRB)});

  NumInstrumentedFrees++;
  return true;
}

bool CilkSanitizerLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  CallGraph *CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();
  auto GetTLI = [this](Function &F) -> TargetLibraryInfo & {
    return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  };
  auto GetDomTree = [this](Function &F) -> DominatorTree & {
    return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  };
  auto GetTaskInfo = [this](Function &F) -> TaskInfo & {
    return this->getAnalysis<TaskInfoWrapperPass>(F).getTaskInfo();
  };
  auto GetLoopInfo = [this](Function &F) -> LoopInfo & {
    return this->getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  };
  auto GetRaceInfo = [this](Function &F) -> RaceInfo & {
    return this->getAnalysis<TapirRaceDetectWrapperPass>(F).getRaceInfo();
  };
  auto GetSE = [this](Function &F) -> ScalarEvolution & {
    return this->getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
  };

  bool Changed =
      CilkSanitizerImpl(M, CG, GetDomTree, nullptr, GetLoopInfo, nullptr,
                        GetTLI, nullptr, CallsMayThrow, JitMode)
          .setup(true);
  Changed |=
      CilkSanitizerImpl(M, CG, GetDomTree, GetTaskInfo, GetLoopInfo,
                        GetRaceInfo, GetTLI, GetSE, CallsMayThrow, JitMode)
          .run();
  return Changed;
}

PreservedAnalyses CilkSanitizerPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto &CG = AM.getResult<CallGraphAnalysis>(M);
  auto GetDT =
    [&FAM](Function &F) -> DominatorTree & {
      return FAM.getResult<DominatorTreeAnalysis>(F);
    };
  auto GetTI =
    [&FAM](Function &F) -> TaskInfo & {
      return FAM.getResult<TaskAnalysis>(F);
    };
  auto GetLI =
    [&FAM](Function &F) -> LoopInfo & {
      return FAM.getResult<LoopAnalysis>(F);
    };
  auto GetRI =
    [&FAM](Function &F) -> RaceInfo & {
      return FAM.getResult<TapirRaceDetect>(F);
    };
  auto GetTLI =
    [&FAM](Function &F) -> TargetLibraryInfo & {
      return FAM.getResult<TargetLibraryAnalysis>(F);
    };
  auto GetSE = [&FAM](Function &F) -> ScalarEvolution & {
    return FAM.getResult<ScalarEvolutionAnalysis>(F);
  };

  bool Changed =
      CilkSanitizerImpl(M, &CG, GetDT, nullptr, GetLI, nullptr, GetTLI, nullptr)
          .setup(false);
  Changed |=
      CilkSanitizerImpl(M, &CG, GetDT, GetTI, GetLI, GetRI, GetTLI, GetSE)
          .run();

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}


// Finds the footprints of F's symbolic nests before the vectorizer changes
// them, and leaves markers for Cilksan (see placeFootprintMarkers). It runs
// at VectorizerStart, after the simplification pipeline (LICM, GVN, full
// unrolling) and before the second LoopRotate, the loop and SLP vectorizers
// and runtime unrolling, so a nest N would cover with constant trips is often
// gone or flat by then. Where VectorizerStart doesn't come before Cilksan (at
// O0, where optnone skips this pass, and in pipelines without it), Cilksan
// places the markers itself, on the IR it sees (see CilkSanitizerImpl::run).
// The attribute marks a function this pass has seen.
PreservedAnalyses CilksanFootprintsPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  if (!(EnableStaticRaceDetection && LoopHoisting && HoistNests &&
        HoistSymbolicNests) ||
      F.empty() || !checkSanitizeCilkAttr(F) ||
      F.hasFnAttribute(FootprintsAttr))
    return PreservedAnalyses::all();
  F.addFnAttr(FootprintsAttr);
  NumFootprintMarkers += CilkSanitizerImpl::placeFootprintMarkers(
      F, AM.getResult<LoopAnalysis>(F), AM.getResult<ScalarEvolutionAnalysis>(F),
      AM.getResult<DominatorTreeAnalysis>(F), AM.getResult<TaskAnalysis>(F));
  // New instructions in preheaders only.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<TaskAnalysis>();
  return PA;
}
