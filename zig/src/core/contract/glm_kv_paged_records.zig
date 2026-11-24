//! GLM's E4M3 latent cache on KV pages (port note af9777b49, after v57 0020): nvcc 13.0's receipts owed, 13.3's in
//! measurements/pending_133.zig. Each paged launch is its window twin's (glm_kv_pending.zig) with a page table:
//! same image, block and shared bytes.
const std = @import("std");
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_lat_write_fp8_paged", .symbol = "tf_glm_lat_write_fp8_paged", .image = "glm_glue", .threads = 512, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_16_fp8_paged", .symbol = "tf_glm_chunks_16_fp8_paged", .image = "glm_glue", .threads = 256, .dynamic = 83328, .opt_in = true, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_32_fp8_paged", .symbol = "tf_glm_chunks_32_fp8_paged", .image = "glm_glue", .threads = 256, .dynamic = 101120, .opt_in = true, .min_sm = 80 }, .usage = registry.none },
};

/// Each paged launch resolves its window twin in glm_kv_pending.zig by the symbol without the `_paged` tail.
fn twin(paged: registry.CudaRecord) !registry.CudaRecord {
    const symbol = paged.launch.symbol[0 .. paged.launch.symbol.len - "_paged".len];
    for (@import("glm_kv_pending.zig").records) |r| if (std.mem.eql(u8, r.launch.symbol, symbol)) return r;
    return error.MissingRecord;
}

test "GLM FP8 KV pages: 13.0 pending, and each launch is its window twin's" {
    const t = std.testing;
    for (cuda) |r| {
        try t.expect(std.mem.endsWith(u8, r.launch.symbol, "_paged"));
        try t.expect(std.mem.endsWith(u8, r.launch.name, r.launch.symbol));
        try t.expectEqual(registry.none, registry.usageFor(r, .{ .major = 13, .minor = 0 }));
        const w = try twin(r);
        try t.expectEqualStrings(w.launch.image, r.launch.image);
        try t.expectEqual(w.launch.threads, r.launch.threads);
        try t.expectEqual(w.launch.dynamic, r.launch.dynamic);
        try t.expectEqual(w.launch.opt_in, r.launch.opt_in);
        try t.expectEqual(w.launch.min_sm, r.launch.min_sm);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
    try t.expectEqual(@as(usize, 3), cuda.len);
}
