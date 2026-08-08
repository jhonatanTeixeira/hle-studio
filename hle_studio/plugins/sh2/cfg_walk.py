"""CFG-following disassembly of one function body - extracted verbatim
(logic unchanged) from jhonatanTeixeira/portal_to_another_world's
tools/study_call.py's `disassemble_function`/`_has_delay_slot`, the
real-play-validated walker MechanicalTranslator (sh2_rust.py) drives.

Never a blind straight-line scan: an earlier straight-line-only version had
a real, concretely-observed bug where an inline literal pool placed mid-
function (reached by falling through in ADDRESS order but never in
EXECUTION order, since every real path jumps over it) got misdecoded as
bogus instructions. This walks a worklist of basic-block start addresses
instead, following BRA/BT/BF/BSR/JSR the way real SH-2 control flow does -
see the docstring below for the exact per-opcode rules.
"""
from __future__ import annotations

from . import sh2dis

MAX_INSNS = 6000  # safety cap: bail out rather than walk forever past a mis-detected function
MAX_BLOCKS = 1500


def _has_delay_slot(mnemonic: str) -> bool:
    if mnemonic in ("RTS", "RTE"):
        return True
    for prefix in ("BRA 0x", "BRAF R", "BSR 0x", "BSRF R", "JSR @R", "JMP @R", "BT/S", "BF/S"):
        if mnemonic.startswith(prefix):
            return True
    return False


def disassemble_function(entry_addr, data, max_insns=None, max_blocks=None):
    """Returns (blocks, all_resolved, note):
      blocks       - list of basic blocks in visitation order (entry block
                     first), each a list of Decoded instructions in address
                     order - concatenating them in this order keeps every
                     branch immediately followed by its own delay-slot
                     instruction, which sh2_to_rust_pseudo.translate_function
                     relies on.
      all_resolved - True if every path cleanly ended at a real RTS/RTE.
      note         - human-readable reason when all_resolved is False.

    Per-opcode rules:
      - RTS/RTE: decode the delay slot, end this path (real function exit).
      - BRA (unconditional, static target): decode delay slot, follow ONLY
        the target - never fall through past it.
      - BT/BF (no delay slot) and BT/S,BF/S (has one): follow BOTH the
        target and the fall-through - both are genuinely reachable.
      - BSR/JSR/BSRF (calls): decode delay slot, continue at the instruction
        AFTER it (a call returns here) - do NOT follow the call target as
        part of THIS function's body.
      - JMP @Rn / BRAF Rn (indirect, no static target): decode delay slot,
        path ends unresolved (no register tracking here) - flagged, not
        hidden.
      - Any address ever seen as a pool_addr is marked as data forever and
        never decoded as a block start.
    """
    max_insns = max_insns if max_insns is not None else MAX_INSNS
    max_blocks = max_blocks if max_blocks is not None else MAX_BLOCKS
    worklist = [entry_addr]
    visited_starts = set()
    pool_addrs = set()
    blocks = []
    total_insns = 0
    unresolved_reasons = []

    while worklist and total_insns < max_insns and len(blocks) < max_blocks:
        start = worklist.pop(0)
        if start in visited_starts or start in pool_addrs:
            continue
        visited_starts.add(start)

        block = []
        pc = start
        while True:
            op = sh2dis.read_u16(data, pc)
            if op is None:
                unresolved_reasons.append(f"address {pc:08X} out of range")
                break
            dec = sh2dis.decode(pc, op, data)
            block.append(dec)
            total_insns += 1
            if dec.pool_addr is not None:
                pool_addrs.add(dec.pool_addr)
            if total_insns >= max_insns:
                unresolved_reasons.append(f"{max_insns}-instruction limit reached")
                break

            m = dec.mnemonic
            if _has_delay_slot(m):
                slot_pc = pc + 2
                op2 = sh2dis.read_u16(data, slot_pc)
                if op2 is not None:
                    dec2 = sh2dis.decode(slot_pc, op2, data)
                    block.append(dec2)
                    total_insns += 1
                    if dec2.pool_addr is not None:
                        pool_addrs.add(dec2.pool_addr)
                after = slot_pc + 2
                if m in ("RTS", "RTE"):
                    pass
                elif m.startswith("BRA 0x"):
                    if dec.target is not None:
                        worklist.append(dec.target)
                    else:
                        unresolved_reasons.append(f"{m} at {pc:08X} has no resolved target")
                elif m.startswith("BRAF"):
                    unresolved_reasons.append(f"BRAF at {pc:08X} - target depends on an untracked register")
                elif m.startswith("BSR 0x") or m.startswith("BSRF") or m.startswith("JSR"):
                    worklist.append(after)
                elif m.startswith("JMP"):
                    unresolved_reasons.append(f"indirect JMP at {pc:08X} - target depends on an untracked register")
                elif m.startswith("BT/S") or m.startswith("BF/S"):
                    if dec.target is not None:
                        worklist.append(dec.target)
                    worklist.append(after)
                break
            elif dec.kind == "cond_branch":
                if dec.target is not None:
                    worklist.append(dec.target)
                worklist.append(pc + 2)
                break
            else:
                pc += 2
        blocks.append(block)

    if total_insns >= max_insns:
        unresolved_reasons.append(f"whole-function {max_insns}-instruction limit reached")
    if len(blocks) >= max_blocks:
        unresolved_reasons.append(f"{max_blocks}-block limit reached")

    all_resolved = len(unresolved_reasons) == 0
    note = "; ".join(unresolved_reasons)
    return blocks, all_resolved, note
