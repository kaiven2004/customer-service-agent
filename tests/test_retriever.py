from __future__ import annotations


def test_policy_question_hits_expected_doc(retriever):
    hits = retriever.search("退货运费谁承担")
    assert hits
    assert hits[0].chunk.doc_id == "returns"
    assert "运费" in hits[0].chunk.heading
    assert retriever.confident(hits)


def test_short_question_finds_refund_policy(retriever):
    hits = retriever.search("退款多久到账")
    assert hits[0].chunk.doc_id == "refund"
    assert retriever.confident(hits)


def test_invoice_question_is_grounded(retriever):
    hits = retriever.search("怎么开发票")
    assert hits[0].chunk.heading == "发票申请"
    assert retriever.confident(hits)


def test_off_topic_question_is_not_confident(retriever):
    hits = retriever.search("今天天气怎么样")
    assert not retriever.confident(hits)


def test_knowledge_base_coverage(retriever):
    queries = {
        "保修多久": {"warranty"},
        "怎么修改收货地址": {"account"},
        "支持哪些支付方式": {"payment"},
        "几天能发货": {"shipping", "faq"},
    }
    for query, doc_ids in queries.items():
        hits = retriever.search(query)
        assert hits, query
        assert hits[0].chunk.doc_id in doc_ids, (query, hits[0].chunk.doc_id)
