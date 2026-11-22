//! A compact prefix of a rank's committed GLM state; live sequence addresses and kernel layouts stay fixed.
const std = @import("std");
const State = @import("cuda_state.zig").State;
const page_rows = @import("cuda_state.zig").page_rows;
const Config = @import("config.zig").Config;
const cache = @import("cuda_dsa_block.zig");
const sparse = @import("cuda_sparse.zig");

const walk = @import("core").state_walk.Device;

/// TF_GLM_KV_LIVE_HALF=1 (checked by cuda_ranks): a keep holds the live KDA half alone; a disk image must name it.
pub fn liveHalfEnabled() bool {
    return std.mem.eql(u8, @import("profile").lookup("TF_GLM_KV_LIVE_HALF") orelse return false, "1");
}

/// The half a plan keeps: `on` with every KDA layer on the same half gives that half; otherwise both (null).
pub fn halfOf(cur: []const u1, layers: usize, on: bool) ?u1 {
    if (!on or layers == 0 or cur.len != layers) return null;
    const h = cur[0];
    for (cur) |b| if (b != h) return null;
    return h;
}

/// `len` bytes from `live`, or on a paged sequence's pages `span` bytes a page `page` bytes apart, from byte `from`;
/// `lat`: a sharded sequence's latent rows (TF_GLM_KV_SHARD), whose pages are its latent table's (`Plan.lids`). Port note (af9777b49): gate 5d.
pub const Part = struct { live: u64, len: usize, span: usize = 0, page: u64 = 0, lat: bool = false, from: usize = 0 };
pub const Plan = struct {
    parts: [5 * (@import("config.zig").max_layers + 1) + 2]Part = undefined,
    count: usize = 0,
    len: usize = 0,
    ids: []const u32 = &.{}, // a paged sequence's pages in row order
    lids: []const u32 = &.{}, // port note (af9777b49): gate 5d — sharded: its latent pages in row order (this rank's, cuda_kv_shard.latentIds)
    sizing: bool = false, // `size`: the rows counted, not read, so no page need be mapped
    latent: bool = true, // the latent codes and scales are parts (sharded: only where this rank keeps them)
    shared: usize = 0, // a paged sequence's first pages a page-sharing store holds: their rows stay on them
    half: ?u1 = null, // TF_GLM_KV_LIVE_HALF: the KDA half the plan carries (null: both, the arena head whole)

    /// The state at `at`; past `shared` pages only, on a paged sequence whose first pages the store holds.
    pub fn init(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize) !Plan {
        return make(st, c, at, head, shared, false, true, halfOf(st.cur, st.kda_layers, liveHalfEnabled())); // port note (af9777b49): gate 5d
    }

    /// A restore's plan: the kept state's own `cur` bits say which half it carries (the slot's may differ by now).
    /// `latent` (false): the sharded keep's plan without its latent rows — the owner copies those alone. Port note (af9777b49): gate 5d.
    pub fn initFor(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize, latent: bool, cur: []const u1) !Plan {
        return make(st, c, at, head, shared, false, latent, halfOf(cur, st.kda_layers, liveHalfEnabled()));
    }

    /// The plan with a given half (tests and tools): both halves for null.
    pub fn withHalf(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize, half: ?u1) !Plan {
        return make(st, c, at, head, shared, false, true, half); // port note (af9777b49): gate 5d
    }

    /// A prefix's bytes at `at`, rows to `capacity` counted held or not (a paged sequence holds its request's alone).
    pub fn size(st: *const State, c: *const Config, capacity: usize, at: usize, head: ?usize, shared: usize) !usize {
        var whole = st.*;
        whole.capacity = capacity;
        return (try make(&whole, c, at, head, shared, true, true, halfOf(st.cur, st.kda_layers, liveHalfEnabled()))).len; // port note (af9777b49): gate 5d
    }

    fn make(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize, sizing: bool, latent: bool, half: ?u1) !Plan {
        if (at > st.capacity or st.caches.len > @import("config.zig").max_layers) return error.SnapshotShape;
        if (st.rings.len != 2 * (st.caches.len + 1)) return error.SnapshotShape;
        if (shared > 0 and (st.table == 0 or shared * page_rows > at)) return error.SnapshotShape; // pages below `at` only
        var p: Plan = .{ .ids = st.pages, .lids = st.lpages, .sizing = sizing, .latent = latent, .shared = shared, .half = half }; // port note (af9777b49): gate 5d
        // a paged sequence's caches are its pages: its fields up to them end at its first ring
        const first = if (st.table != 0) st.rings[0] else if (st.caches.len > 0) st.caches[0].lc else st.mtp_caches.lc;
        if (first < st.arena.buf.ptr) return error.SnapshotShape;
        if (half) |h| {
            // the arena head in two parts around the dead half: [start, dead) and [dead + len, first)
            const len = try std.math.mul(usize, st.kda_layers, st.rec_layer);
            const dead = st.rec + @as(u64, 1 - h) * len;
            if (dead < st.arena.buf.ptr or dead > first or first - dead < len) return error.SnapshotShape;
            try p.add(st, st.arena.buf.ptr, @intCast(dead - st.arena.buf.ptr));
            try p.add(st, dead + len, @intCast(first - (dead + len)));
        } else try p.add(st, st.arena.buf.ptr, @intCast(first - st.arena.buf.ptr));
        for (st.caches, 0..) |x, j| try p.rows(st, c, x, j, at);
        if (head) |n| {
            if (n > st.capacity) return error.SnapshotShape;
            try p.rows(st, c, st.mtp_caches, st.caches.len, n);
        }
        if (st.mtp_pos_dev < st.arena.buf.ptr or st.mtp_pos_dev - st.arena.buf.ptr > st.arena.buf.len)
            return error.SnapshotShape;
        const end = if (st.table != 0) st.table else st.arena.buf.ptr + st.arena.buf.len; // its table stays its own
        if (end < st.mtp_pos_dev) return error.SnapshotShape;
        try p.add(st, st.mtp_pos_dev, @intCast(end - st.mtp_pos_dev));
        return p;
    }

    fn rows(p: *Plan, st: *const State, c: *const Config, x: cache.Caches, j: usize, at: usize) !void {
        const latent = try std.math.mul(usize, at, @as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1));
        const ring = try std.math.mul(usize, @min(at, st.ring_rows), st.ring_row);
        const pooled = try std.math.mul(usize, @min(at / sparse.pool + 2, x.pk_rows), @as(usize, c.i_dim) * 2);
        const latent_row = @as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1);
        if (p.latent) { // port note (af9777b49): gate 5d — the sharded latent rows live on this rank's latent pages alone
            try p.addLatent(st, x, x.lc, latent, page_rows * latent_row);
            // E4M3 scales: beside the codes in the arena, or on pages a part's 512 after its codes
            if (x.ls != 0) try p.addLatent(st, x, x.ls, try std.math.mul(usize, at, @sizeOf(f32)), page_rows * @sizeOf(f32));
        } else {
            try p.addCache(st, x, x.lc, latent, page_rows * @as(usize, c.kv_lora) * 2, x.ps);
            if (x.ls != 0) try p.add(st, x.ls, try std.math.mul(usize, at, @sizeOf(f32)));
        }
        try p.add(st, st.rings[2 * j], ring);
        try p.add(st, st.rings[2 * j + 1], ring);
        try p.addCache(st, x, x.pk, pooled, page_rows / sparse.pool * @as(usize, c.i_dim) * 2, if (x.pks != 0) x.pks else x.ps);
    }

    /// A cache's rows: in the arena, or on the sequence's pages, `span` bytes a page (past the store's `shared`).
    fn addCache(p: *Plan, st: *const State, x: cache.Caches, live: u64, len: usize, span: usize, page: u64) !void {
        if (x.pt == 0) return p.add(st, live, len);
        const from = @min(len, p.shared * span);
        if (len == from) return;
        if ((!p.sizing and (len - 1) / span >= p.ids.len) or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len - from);
        p.parts[p.count] = .{ .live = live, .len = len - from, .span = span, .page = page, .from = from };
        p.count += 1;
    }

    /// A sharded cache's latent rows: on this rank's latent pages (`x.lpt`'s table, its `Plan.lids`), or in the arena
    /// without them. Port note (af9777b49): gate 5d — only this rank's latent rows are parts of the plan.
    fn addLatent(p: *Plan, st: *const State, x: cache.Caches, live: u64, len: usize, span: usize) !void {
        if (x.lpt == 0) return p.addCache(st, x, live, len, span, x.ps); // the arena's own rows, unsharded
        const from = @min(len, p.shared * span); // the store's pages keep their latent rows too (a part's pages)
        if (len == from) return;
        if ((!p.sizing and (len - 1) / span >= p.lids.len) or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len - from);
        p.parts[p.count] = .{ .live = live, .len = len - from, .span = span, .page = x.lps, .lat = true, .from = from };
        p.count += 1;
    }

    /// Where a part's bytes from `done` on sit, as many as lie in a row there.
    pub fn piece(p: *const Plan, part: Part, done: usize) Part {
        if (part.span == 0) return .{ .live = part.live + done, .len = part.len - done }; // port note (af9777b49): gate 5d — the lat parts walk the latent ids
        const at = part.from + done;
        const off = at % part.span;
        const ids = if (part.lat) p.lids else p.ids;
        return .{ .live = part.live + @as(u64, ids[at / part.span]) * part.page + off, .len = @min(part.span - off, part.len - done) };
    }

    fn add(p: *Plan, st: *const State, live: u64, len: usize) !void {
        if (len == 0) return;
        if (live < st.arena.buf.ptr) return error.SnapshotShape;
        const off = live - st.arena.buf.ptr;
        if (off > st.arena.buf.len or len > st.arena.buf.len - off or p.count == p.parts.len)
            return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len };
        p.count += 1;
    }
};
