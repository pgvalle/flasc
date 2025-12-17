from flwr.serverapp.strategy.fedavg import *
import sys

def my_aggregate_arrayrecords(
        records: list[RecordDict], weighting_metric_name: str
) -> ArrayRecord:
    """Perform weighted aggregation all ArrayRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        # Because replies have been checked for consistency,
        # we can safely cast the weighting factor to float
        w = cast(float, metricrecord[weighting_metric_name])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    # Perform weighted aggregation
    aggregated_np_arrays: dict[str, NDArray] = {}

    for record, weight in zip(records, weight_factors, strict=True):
        for record_item in record.array_records.values():
            # aggregate in-place
            for key, value in record_item.items():
                if key not in aggregated_np_arrays:
                    aggregated_np_arrays[key] = value.numpy() * weight
                else:
                    aggregated_np_arrays[key] += value.numpy() * weight

    return ArrayRecord(
            {k: Array(np.asarray(v)) for k, v in aggregated_np_arrays.items()}
            )


class FLASC(FedAvg):

    def __init__(self, fraction_train=1.0) -> None:
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

            #!
            # Aggregate ArrayRecords
            arrays = my_aggregate_arrayrecords(
                    reply_contents,
                    self.weighted_by_key,
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                    reply_contents,
                    self.weighted_by_key,
                    )
        return arrays, metrics
