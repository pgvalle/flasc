from flwr.serverapp.strategy import FedAvg
from flwr.common import (
    ArrayRecord,
    MetricRecord,
    RecordDict,
    Array,
    Message,
    log,
)

from logging import INFO, WARNING
from typing import Iterable


METRICRECORD_WEIGHT_KEY = "num_samples"
METRICRECORD_IDENTITIES_KEY = "identities"

ARRAYRECORD_KEY = "arrays"
CONFIGRECORD_KEY = "config"
METRICRECORD_KEY = "metrics"


class FLASC(FedAvg):

    def __init__(
            self,
            num_models: int,
            fraction_train: float = 0.5,
            fraction_evaluate: float = 1.0,
            min_train_nodes: int = 2,
            min_evaluate_nodes: int = 2,
            min_available_nodes: int = 2,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=METRICRECORD_WEIGHT_KEY,
            arrayrecord_key=ARRAYRECORD_KEY,
            configrecord_key=CONFIGRECORD_KEY,
        )
        self.num_models = num_models


    @staticmethod
    def pack_arrays_list(arrays_list: list[ArrayRecord]) -> ArrayRecord:
        """Pack a list of array records into a single array record"""
    
        packed = ArrayRecord()
        i = 0
        for arrays in arrays_list:
            for key, array in arrays.items():
                indexed_key = f"{i} {key}"
                packed[indexed_key] = array
            i += 1
        return packed


    def unpack_arrays(arrays: ArrayRecord) -> list[ArrayRecord]:
        """Unpack an array record into a list of array records"""

        arrays_list = []
        for indexed_key, array in arrays.items():
            i_str, key = indexed_key.split()
            i = int(i_str)

            while len(arrays_list) <= i:
                arrays_list.append(ArrayRecord())

            arrays_list[i][key] = array

        return arrays_list


    @staticmethod
    def update_clusters(
            records: list[RecordDict],
            weighting_metric_name: str,
            num_global_models: int,
    ) -> ArrayRecord:
        """Update each cluster model"""

        # Group RecordDicts to form the clusters
        clusters = [[] for _ in range(num_global_models)]
        for record in records:
            metrics = next(iter(record.metric_records.values()))
            identities = metrics[METRICRECORD_IDENTITIES_KEY]
            # add this client RecordDict to all clusters it belongs to
            for i in identities:
                clusters[i].append(record)

        from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

        # Do FedAvg on each cluster
        arrays_list = []
        for cluster in clusters:
            arrays = aggregate_arrayrecords(cluster, weighting_metric_name)
            arrays_list.append(arrays)

        # pack arrays_list into a single array and return
        return FLASC.pack_arrays_list(arrays_list)


    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""

        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate ArrayRecords
            arrays = FLASC.update_clusters(
                reply_contents,
                self.weighted_by_key,
                self.num_models,
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return arrays, metrics


    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate MetricRecords in the received Messages."""

        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return metrics
