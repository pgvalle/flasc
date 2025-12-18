from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
from flwr.common import (
    ArrayRecord,
    MetricRecord,
    RecordDict,
    Array,
    Message,
    log,
)

from flasc.utils import pack_state_dicts

from typing import Iterable


def my_aggregate_arrayrecords(
        records: list[RecordDict],
        weighting_metric_name: str,
        num_global_models: int
) -> ArrayRecord:
    """Perform weighted aggregation all ArrayRecords using a specific key."""

    # Group RecordDicts to form the clusters
    clusters = [[] for _ in range(num_global_models)]
    for record in records:
        config = next(iter(record.config_records.values()))
        identities = config["identities"]
        # add this client RecordDict to all clusters it belongs to
        for i in identities:
            clusters[i].append(record)
        
    # may cause bugs on record not being copied (reference)
    # Do FedAvg on each cluster separately
    state_dicts = []
    for cluster in clusters:
        state_dict = aggregate_arrayrecords(cluster, weighting_metric_name)
        state_dicts.append(state_dict)

    # pack models and return them
    return pack_state_dicts(state_dicts)


class FLASC(FedAvg):

    def __init__(self, num_models, fraction_train=1.0) -> None:
        self.num_models = num_models
        super().__init__(fraction_train=fraction_train)


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
            arrays = my_aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
                self.num_models,
            )

            #! ignoring Aggregate MetricRecords for now
            # metrics = self.train_metrics_aggr_fn(
            #     reply_contents,
            #     self.weighted_by_key,
            # )
        return arrays, MetricRecord()
