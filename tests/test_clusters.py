"""Identity clustering over the FF++ pair graph, and crop-name parsing.

These underpin every cluster-aware confidence interval, so a silent bug here
would quietly invalidate the paper's inference rather than crash.
"""

from src.clusters import build_identity_clusters, cluster_stats
from src.predict import parse_crop_name


def test_reverse_pair_lands_in_one_component():
    """The exact leak video-level grouping misses: 004_982 and 982_004 group
    separately by target, but share both identities."""
    clusters = build_identity_clusters([("004", "982"), ("982", "004")])
    assert clusters["004"] == clusters["982"]


def test_transitive_identities_merge():
    # 1-2, 2-3 => {1,2,3} one component even though 1 and 3 never co-occur
    clusters = build_identity_clusters([("1", "2"), ("2", "3")])
    assert clusters["1"] == clusters["2"] == clusters["3"]


def test_disjoint_pairs_stay_separate():
    clusters = build_identity_clusters([("1", "2"), ("3", "4")])
    assert clusters["1"] == clusters["2"]
    assert clusters["3"] == clusters["4"]
    assert clusters["1"] != clusters["3"]


def test_original_with_no_source_is_a_singleton():
    clusters = build_identity_clusters([("7", "")])
    assert clusters["7"] == "7"
    assert cluster_stats(clusters)["n_components"] == 1


def test_component_id_is_stable_regardless_of_input_order():
    a = build_identity_clusters([("004", "982"), ("982", "004")])
    b = build_identity_clusters([("982", "004"), ("004", "982")])
    assert a == b


def test_stats_counts():
    clusters = build_identity_clusters([("1", "2"), ("3", "4"), ("5", "")])
    s = cluster_stats(clusters)
    assert s["n_sequences"] == 5
    assert s["n_components"] == 3
    assert s["singletons"] == 1


def test_parse_fake_crop_name():
    m = parse_crop_name(
        "/x/fake/manipulated-sequences-Deepfakes-c40-videos-004-982_00012.jpg")
    assert m["manipulation"] == "Deepfakes"
    assert m["compression"] == "c40"
    assert m["target_seq"] == "004"
    assert m["source_seq"] == "982"
    assert m["video_id"] == "004"
    assert m["frame_idx"] == 12


def test_parse_real_crop_name():
    m = parse_crop_name(
        "/x/real/original-sequences-youtube-c23-videos-602_00036.jpg")
    assert m["manipulation"] == "real"
    assert m["compression"] == "c23"
    assert m["target_seq"] == "602"
    assert m["source_seq"] == ""
    assert m["frame_idx"] == 36


def test_video_id_matches_the_group_by_regex_capture():
    """`--group-by 'videos-([0-9]+)'` captures the target sequence; parse_crop_name
    must agree or splits and clusters would disagree about what a group is."""
    import re
    for path in [
        "/x/fake/manipulated-sequences-NeuralTextures-c23-videos-004-982_00000.jpg",
        "/x/real/original-sequences-youtube-c23-videos-602_00036.jpg",
    ]:
        regex_capture = re.search(r"videos-([0-9]+)", path).group(1)
        assert parse_crop_name(path)["video_id"] == regex_capture


def test_ffpp_pair_graph_gives_size_two_components_not_one_giant():
    """Real FF++ pairs are disjoint couples, so identity clustering yields many
    small components. A cyclic pair set (a->b->c->a) would instead collapse
    everything into one component and make the identity bootstrap degenerate —
    this test pins the distinction."""
    disjoint = [("004", "982"), ("982", "004"), ("111", "222"), ("222", "111")]
    c = build_identity_clusters(disjoint)
    assert cluster_stats(c)["n_components"] == 2
    assert cluster_stats(c)["largest"] == 2

    cyclic = [("1", "2"), ("2", "3"), ("3", "1")]
    assert cluster_stats(build_identity_clusters(cyclic))["n_components"] == 1


def test_identity_clustering_halves_units_relative_to_video_grouping():
    """Each FF++ couple contributes two target sequences (a_b and b_a), so
    grouping by identity yields half as many independent units as grouping by
    target video. This is why intervals widen at the stricter unit."""
    pairs = [("004", "982"), ("982", "004"), ("111", "222"), ("222", "111")]
    targets = {t for t, _ in pairs}
    comps = set(build_identity_clusters(pairs).values())
    assert len(targets) == 4
    assert len(comps) == 2


def test_all_four_ffpp_methods_parse_including_digits_in_the_name():
    """Face2Face contains a digit. An [A-Za-z]+ method class silently failed to
    match it, leaving 600 rows with an empty video_id that would have collapsed
    into one bogus cluster and corrupted every downstream interval. Regression
    test for that bug."""
    cases = [
        ("manipulated-sequences-Deepfakes-c40-videos-004-982_00012.jpg", "Deepfakes"),
        ("manipulated-sequences-Face2Face-c40-videos-004-982_00012.jpg", "Face2Face"),
        ("manipulated-sequences-FaceSwap-c23-videos-111-222_00000.jpg", "FaceSwap"),
        ("manipulated-sequences-NeuralTextures-raw-videos-7-8_00036.jpg", "NeuralTextures"),
    ]
    for path, method in cases:
        m = parse_crop_name(path)
        assert m["manipulation"] == method, (path, m)
        assert m["video_id"], f"empty video_id for {path}"
        assert m["source_seq"], f"empty source_seq for {path}"


def test_no_crop_name_yields_an_empty_video_id():
    """Any unparsed path becomes a cluster of its own with an empty key, which
    silently merges unrelated items. Nothing should reach that fallback."""
    for path in [
        "manipulated-sequences-Face2Face-c40-videos-004-982_00012.jpg",
        "original-sequences-youtube-c40-videos-602_00036.jpg",
    ]:
        assert parse_crop_name(path)["video_id"] != ""
