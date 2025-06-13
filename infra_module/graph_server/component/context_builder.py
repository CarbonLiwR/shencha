import json
import pandas as pd
from typing import cast, Any
from ..util import get_entity_information_by_id, num_tokens, remove_unrelated_attributes
from .ranker import FieldRanker

class GraphContextBuilder():
    def __init__(self, max_context_tokens, ranker: FieldRanker):
        self.max_context_tokens = max_context_tokens
        self.ranker: FieldRanker = ranker

    def __ranking_data(self, data):
        ranked_data = self.ranker.rank(data)
        return ranked_data

    def __build_entity_context(self, graph):
        current_context_text = f"-----Entities-----" + "\n"
        current_token = 0

        # 添加表头
        header = ["id", "entity_type", "entity_name", "text"]
        current_context_text += "|".join(header) + "\n"
        all_context_records = [header]

        # 依据匹配的实体列表中实体的顺序进行排序
        for id, entity in enumerate(graph.entities):
            attributes = " ".join(
                [f"{k}: {v}" for k, v in remove_unrelated_attributes(entity.attributes).items()])
            new_context = [
                str(id),
                entity.type,
                entity.name,
                attributes
            ]

            new_context_text = "|".join(new_context) + "\n"
            current_context_text += new_context_text
            all_context_records.append(new_context)
            current_token += num_tokens(current_context_text)
            if current_token >= self.max_context_tokens:
                break

        if len(all_context_records) > 1:
            record_df = pd.DataFrame(
                all_context_records[1:], columns=cast(Any, all_context_records[0])
            )
        else:
            record_df = pd.DataFrame()

        return current_context_text, record_df

    def __build_relationship_context(self, graph):
        self.ranker.init_field(lambda x: x.attributes["links"] if "links" in x.attributes else float("inf"))
        graph.relationship_instances = self.__ranking_data(graph.relationship_instances)
        # 添加上下文标题
        current_context_text = f"-----Relationships-----" + "\n"
        current_token = 0

        # 添加表头
        header = ["id", "name", "source", "target", "text"]
        current_context_text += "|".join(header) + "\n"
        all_context_records = [header]

        # 添加关系数据
        for id, relationship in enumerate(graph.relationship_instances):
            source_entity_name = next(
                (entity.name for entity in graph.entities if str(entity.id) == relationship.source_entity_uuid),
                None)
            target_entity_name = next(
                (entity.name for entity in graph.entities if str(entity.id) == relationship.target_entity_uuid),
                None)
            source_entity = get_entity_information_by_id(graph.entities, relationship.source_entity_uuid)
            target_entity = get_entity_information_by_id(graph.entities, relationship.target_entity_uuid)
            attributes = f"({source_entity}, {relationship.name}, {target_entity})"
            new_context = [
                str(id),
                relationship.name,
                source_entity_name,
                target_entity_name,
                attributes
            ]

            new_context_text = "|".join(new_context) + "\n"
            current_context_text += new_context_text
            all_context_records.append(new_context)
            current_token += num_tokens(current_context_text)
            if current_token >= self.max_context_tokens:
                break

        if len(all_context_records) > 1:
            record_df = pd.DataFrame(
                all_context_records[1:], columns=cast(Any, all_context_records[0])
            )
        else:
            record_df = pd.DataFrame()

        return current_context_text, record_df

    def __build_community_context(self, graph):
        self.ranker.init_field(lambda x: x.attributes["links"] if "links" in x.attributes else float("inf"))
        graph.community_reports = self.__ranking_data(graph.community_reports)

        def _is_included(report) -> bool:
            return report.rating is not None and float(report.rating) >= 1

        selected_reports = [report for report in graph.community_reports if _is_included(report)]
        if not selected_reports:
            return "", pd.DataFrame()

        current_context_text = f"-----Reports-----\n"
        current_token = 0

        # 添加表头
        header = ["id", "text"]
        current_context_text += "|".join(header) + "\n"
        all_context_records = [header]

        # 构建上下文文本和记录
        for id, report in enumerate(selected_reports):
            new_context = [
                str(id),
                "\n".join([
                    f"社区名称: {json.loads(report.content)['title']}",
                    f"社区摘要: {json.loads(report.content)['summary']}",
                    *[
                        f"  详细摘要({index + 1}): {finding['summary']}\n 摘要说明({index + 1}): {finding['explanation']}"
                        for index, finding in enumerate(json.loads(report.content)["findings"])
                    ]
                ])
            ]
            new_context_text = "|".join(new_context) + "\n"
            current_context_text += new_context_text
            all_context_records.append(new_context)
            current_token += num_tokens(current_context_text)
            if current_token >= self.max_context_tokens:
                break

        if len(all_context_records) > 1:
            record_df = pd.DataFrame(all_context_records[1:], columns=all_context_records[0])
        else:
            record_df = pd.DataFrame()

        return current_context_text, record_df

    def __build_triple_source_context(self, graph):
        if graph.relationship_instances is None or len(graph.relationship_instances) == 0:
            return "", ""
        # 添加上下文标题
        current_context_text = f"-----Sources-----" + "\n"
        current_token = 0

        # 添加表头
        header = ["id", "text"]
        current_context_text += "|".join(header) + "\n"
        all_context_records = [header]

        select_triple_sources = set()
        select_triple_sources.update(
            relationship.triple_source
            for relationship in graph.relationship_instances
            if relationship.triple_source
        )

        # 提取每个实体对应的三元组的信息源
        for id, unit in enumerate(select_triple_sources):
            new_context = [
                str(id),
                unit
            ]
            new_context_text = "|".join(new_context) + "\n"

            current_context_text += new_context_text
            all_context_records.append(new_context)
            current_token += num_tokens(current_context_text)
            if current_token >= self.max_context_tokens:
                break

        if len(all_context_records) > 1:
            record_df = pd.DataFrame(
                all_context_records[1:], columns=cast(Any, all_context_records[0])
            )
        else:
            record_df = pd.DataFrame()

        return current_context_text, record_df

    def build_context(self, graph):
        entity_context_str, entity_context_df = self.__build_entity_context(graph)
        relationship_context_str, relationship_context_df = self.__build_relationship_context(graph)
        community_context_str, community_context_df = self.__build_community_context(graph)
        triple_source_context_str, triple_source_context_df = self.__build_triple_source_context(graph)
        context_str = entity_context_str + relationship_context_str + community_context_str + triple_source_context_str
        context_df = {
            "entity": entity_context_df,
            "relationship": relationship_context_df,
            "community": community_context_df,
            "triple_source": triple_source_context_df,
        }
        return context_str, context_df