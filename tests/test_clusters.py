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
