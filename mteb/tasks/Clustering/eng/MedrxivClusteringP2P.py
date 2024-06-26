from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


class MedrxivClusteringP2PFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="MedrxivClusteringP2P.v2",
        description="Clustering of titles+abstract from medrxiv across 51 categories.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "mteb/medrxiv-clustering-p2p",
            "revision": "9894e30672c61db02f10a8593519d84e2b7a1a1c",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        form=["written"],
        domains=["Academic", "Medical"],
        task_subtypes=["Thematic clustering"],
        license="https://www.medrxiv.org/content/about-medrxiv",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="",
        n_samples={"test": 2048},
        avg_character_length={"test": 1984.7},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=2048,
        )


class MedrxivClusteringP2P(AbsTaskClustering):
    superseeded_by = "MedrxivClusteringP2P.v2"
    metadata = TaskMetadata(
        name="MedrxivClusteringP2P",
        description="Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "mteb/medrxiv-clustering-p2p",
            "revision": "e7a26af6f3ae46b30dde8737f02c07b1505bcc73",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Thematic clustering"],
        license="https://www.medrxiv.org/content/about-medrxiv",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="",
        n_samples={"test": 375000},
        avg_character_length={"test": 1981.2},
    )
