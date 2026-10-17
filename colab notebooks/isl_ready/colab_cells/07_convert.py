# CONVERT — pilot is already built; convert the split shards in archive order (resumable, disk-guarded, per-shard timing).
CONVERT_INFO = {}
for _split in SPLITS:
    limit = None if FULL_CONVERT else CONVERT_MAX_SHARDS[_split]
    print(f"\n{_split}: {len(rows_of[_split]):,} planned sequences, converting {'all' if limit is None else limit} shard(s) of {SHARD_SIZE}")
    CONVERT_INFO[_split] = ip.convert_manifest(convert_rows_of[_split], PROC_ROOT / _split, read_pose_by_uid, NORM, QUALITY,
                                               SHARD_SIZE, max_shards=limit, min_free_gb=MIN_FREE_GB, disk_path=DISK_GUARD_PATH)
    if CONVERT_INFO[_split]["stopped"]:
        disk_report()
        raise RuntimeError(f"Conversion of '{_split}' stopped: {CONVERT_INFO[_split]['stopped']}. "
                           "Finished shards are kept. Free space, then re-run this cell to resume.")

STORES = {s: ip.ShardStore(PROC_ROOT / s, expected_hash=NORM_HASH) for s in SPLITS}
conv_rows, exclusion_rows = [], []
for s, st in STORES.items():
    exc = st.exclusions()
    exclusion_rows += [{"split": s, **e} for e in exc]
    conv_rows.append({"split": s, "planned": len(rows_of[s]), "shards": len(st.metas), "kept": len(st), "excluded": len(exc),
                      "videos_kept": len({ip.video_id_of(u) for u in st.uids}), "frames": int(st.lengths.sum()),
                      "mean_valid_fraction": float(st.valid_fraction.mean()) if len(st) else float("nan"),
                      "non_300x300_sequences": sum(1 for d in st.dims if d is not None and d != (300.0, 300.0))})
display(pd.DataFrame(conv_rows))
if exclusion_rows:
    exc_df = pd.DataFrame(exclusion_rows)
    exc_df["reason_group"] = exc_df["reason"].str.replace(r"\s*\(.*", "", regex=True)
    display(exc_df.groupby(["split", "reason_group"]).size().rename("count").reset_index())
    exc_df.drop(columns="reason_group").to_csv(LOG_DIR / "excluded_sequences.csv", index=False)
    print("every excluded sequence is listed in", LOG_DIR / "excluded_sequences.csv")
else:
    print("no sequences were excluded")
