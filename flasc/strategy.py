from flwr.serverapp.strategy import FedAvg, Result
from flwr.common import (
    ArrayRecord,
    MetricRecord,
    ConfigRecord,
    RecordDict,
    Message,
    log,
)

from logging import INFO, WARNING
from typing import Iterable, Callable, cast

from matplotlib.streamplot import Grid


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
            train_metrics_aggr_fn = FLASC.aggregate_metrics,
            evaluate_metrics_aggr_fn = FLASC.aggregate_metrics
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
                packed.update({indexed_key: array})
            i += 1
        return packed


    @staticmethod
    def unpack_arrays(arrays: ArrayRecord) -> list[ArrayRecord]:
        """Unpack an array record into a list of array records"""

        arrays_list = []
        for indexed_key, array in arrays.items():
            i_str, key = indexed_key.split()
            i = int(i_str)

            while len(arrays_list) <= i:
                arrays_list.append(ArrayRecord())

            arrays_list[i].update({key: array})

        return arrays_list


    def update_clusters(
            self,
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

        # Do FedAvg on each cluster that has members
        for i in range(num_global_models):
            if clusters[i] == []:
                log(INFO, f"cluster {i} has no update!")
                continue
            self.list_arrays[i] = aggregate_arrayrecords(clusters[i], weighting_metric_name)

        # pack arrays_list into a single array and return
        return FLASC.pack_arrays_list(self.list_arrays)


    @staticmethod
    def aggregate_metrics(
            records: list[RecordDict], weighting_metric_name: str
    ) -> MetricRecord:
        """Perform weighted aggregation all MetricRecords using a specific key."""

        weights = []
        for record in records:
            # Get the first (and only) MetricRecord in the record
            metrics = next(iter(record.metric_records.values()))
            # Because replies have been checked for consistency,
            # we can safely cast the weighting factor to float
            w = cast(float, metrics[weighting_metric_name])
            weights.append(w)

        # Average
        total_weight = sum(weights)
        weight_factors = [w / total_weight for w in weights]

        aggregated_metrics = MetricRecord()
        for record, weight in zip(records, weight_factors, strict=True):
            for record_item in record.metric_records.values():
                # aggregate in-place
                for key, value in record_item.items():
                    if key in [weighting_metric_name, METRICRECORD_IDENTITIES_KEY, "identity"]:
                        # We exclude the weighting key and identity vector from the aggregated MetricRecord
                        continue
                    if key not in aggregated_metrics:
                        aggregated_metrics[key] = value * weight
                    else:
                        current_value = cast(float, aggregated_metrics[key])
                        aggregated_metrics[key] = current_value + value * weight
    
        return aggregated_metrics

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
            arrays = self.update_clusters(
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

    def start(
            self,
            grid: Grid,
            initial_list_arrays: list[ArrayRecord],
            num_rounds: int = 3,
            timeout: float = 3600,
            train_config: ConfigRecord | None = None,
            evaluate_config: ConfigRecord | None = None,
            evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        self.list_arrays = initial_list_arrays

        initial_arrays = FLASC.pack_arrays_list(initial_list_arrays)
        return super().start(
            grid,
            initial_arrays,
            num_rounds,
            timeout,
            train_config,
            evaluate_config,
            evaluate_fn
        )