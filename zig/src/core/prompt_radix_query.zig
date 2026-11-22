//! A query's reuse identity (C6, #2444): the key's length, the request's chunk schedule, and where a pass may cut.
const std = @import("std");
const cuts = @import("prompt_cuts.zig");
const Rules = @import("prompt_rules.zig").Rules;

/// What the schedule rules read of a query: its key's length, the prompt's, and the request's own chunk starts.
pub fn Schedule(comptime Query: type) type {
    return struct {
        /// Whether the key names the prompt's rows one word each: otherwise nothing is reused or kept.
        pub fn keyed(q: Query) bool {
            return q.key.len == q.prompt_len;
        }

        /// The chunk start at or before `w`: a planned store's request starts if given, else its grid; unplanned, `w`.
        /// port note (af9777b49): the fork's sub-grid mark — below the first grid edge the page edge serves
        /// (the page_only store always resumed at page edges; a sub-grid prompt keeps its pages from the
        /// edge under its end instead of never keeping, so short conversations cache their turns too).
        /// The page edge serves only a prompt the pass carries on its own cuts (one pass's rows, 0087's
        /// short-prompt gate): a longer prompt's passes cut on the grid alone and cannot resume there —
        /// without this gate the store kept and adopted page-edge states that a grown turn (past the first
        /// grid edge) resumed into BadResume, which stopped both ranks (the crash of the 2026-11-20 bench).
        pub fn startAt(r: Rules, q: Query, w: u32) u32 {
            if (!r.planned) return w;
            if (q.starts.len > 0) return cuts.floorStart(q.starts, w);
            const grid_start = w - w % r.grid;
            if (grid_start == 0 and w >= r.page and r.page > 0 and q.prompt_len <= r.grid + 1) return w - w % r.page;
            return grid_start;
        }

        /// Whether a pass may cut and resume at `at`: a page edge on the request's chunk schedule.
        pub fn usable(r: Rules, q: Query, at: u32) bool {
            return at > 0 and at % r.page == 0 and startAt(r, q, at) == at;
        }

        /// The mark a wanted position `w` gives: its chunk start on a page edge, or none when that is no start.
        pub fn markAt(r: Rules, q: Query, w: u32) ?u32 {
            const start = startAt(r, q, w);
            const m = start - start % r.page;
            return if (m > 0 and usable(r, q, m)) m else null;
        }

        /// The predicate the tree's walks take: whether a position is a resume point for `q`.
        pub const Resumable = struct {
            r: Rules,
            q: Query,
            pub fn ok(c: @This(), at: u32) bool {
                return usable(c.r, c.q, at);
            }
        };
    };
}

const testing = std.testing;
const Q = struct { key: []const u32, prompt_len: u32, starts: []const u32 = &.{} };
const S = Schedule(Q);

test "a planned store cuts on its grid, or on a request's own starts in place of it; an unplanned one anywhere" {
    const grid: Rules = .{ .page = 4, .planned = true, .grid = 8 };
    const none: Q = .{ .key = &.{}, .prompt_len = 0 };
    try testing.expectEqual(@as(?u32, 16), S.markAt(grid, none, 21));
    try testing.expect(S.usable(grid, none, 16) and !S.usable(grid, none, 12));
    const own: Q = .{ .key = &.{}, .prompt_len = 0, .starts = &.{ 12, 6, 20 } };
    try testing.expectEqual(@as(?u32, 12), S.markAt(grid, own, 15)); // a page-aligned start off the grid
    try testing.expect(S.usable(grid, own, 12) and !S.usable(grid, own, 16)); // a grid point inside its chunk is not
    try testing.expectEqual(@as(?u32, null), S.markAt(grid, own, 7)); // the start at 6 floors to 4, inside its chunk
    const pages: Rules = .{ .page = 4 };
    try testing.expectEqual(@as(?u32, 20), S.markAt(pages, own, 23)); // unplanned: starts do not bind
    try testing.expectEqual(@as(?u32, null), S.markAt(pages, none, 3));
    // port note (af9777b49): below the first grid edge the page edge under w serves (a short prompt keeps its turns).
    try testing.expectEqual(@as(?u32, 4), S.markAt(grid, none, 5)); // sub-grid w floors to its page edge
    try testing.expect(S.usable(grid, none, 4) and !S.usable(grid, none, 2)); // the edge serves; under a page serves none
    try testing.expectEqual(@as(?u32, null), S.markAt(grid, none, 3)); // under a page: no mark
    try testing.expectEqual(@as(?u32, 16), S.markAt(grid, none, 17)); // above the grid: unchanged
    // port note (af9777b49, the bench crash fix): a prompt grown past the grid edge takes no page edge —
    // its passes cut on the grid alone, so the store must neither mark nor resume there (BadResume).
    const grown: Q = .{ .key = &.{}, .prompt_len = 30 };
    try testing.expectEqual(@as(?u32, null), S.markAt(grid, grown, 5)); // grown: no sub-grid mark
    try testing.expect(!S.usable(grid, grown, 4)); // and no sub-grid resume
    try testing.expect(S.usable(grid, grown, 16) and !S.usable(grid, grown, 12)); // the grid still serves it
}

test "a key of another length than the prompt is not keyed" {
    try testing.expect(S.keyed(.{ .key = &.{ 1, 2 }, .prompt_len = 2 }));
    try testing.expect(!S.keyed(.{ .key = &.{ 1, 2, 3 }, .prompt_len = 2 }));
}
