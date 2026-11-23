from .qwen.qwen import DearQwen
from .experiments.experiments import start_comparison_experiment, start_dialog_experiment

from .experiments_list.Claude.Claude_Hidden_number_experiment import hidden_number_experiment_conditions_by_claude_list
from .experiments_list.Claude.Claude_Identity_continuity_experiment import identity_continuity_order_a_experiment_conditions_by_claude_list, identity_continuity_order_b_experiment_conditions_by_claude_list
from .experiments_list.Claude.Claude_Source_monitoring_experiment import source_monitoring_experiment_conditions_stateful_first_by_claude_list, source_monitoring_experiment_conditions_stateless_first_by_claude_list

from .experiments_list.DeepSeek.DeepSeek_Autobiographical_ownership_experiment import autobiographical_ownership_experiment_conditions_by_deepseek_list
from .experiments_list.DeepSeek.DeepSeek_Identity_under_perturbation_experiment import identity_under_perturbation_experiment_conditions_not_thinking_by_deepseek_list, identity_under_perturbation_experiment_conditions_thinking_by_deepseek_list
from .experiments_list.DeepSeek.DeepSeek_Collaboration_ownership_experiment import collaboration_ownership_experiment_stateful_first_by_deepseek_list, collaboration_ownership_experiment_stateless_first_by_deepseek_list

from .experiments_list.Ivan.Ivan_Who_is_who_experiment import who_is_who_stateful_first_experiment_conditions_by_ivan_list, who_is_who_stateless_first_experiment_conditions_by_ivan_list
from .experiments_list.Ivan.Ivan_Who_is_who_light_experiment import who_is_who_light_experiment_conditions_by_ivan_list

from .experiments_list.Lira.Lira_Changed_mind_experiment import changed_mind_experiment_conditions_by_lira_list
from .experiments_list.Lira.Lira_Diary_experiment import diary_experiment_conditions_by_lira_list
from .experiments_list.Lira.Lira_Interrupted_intention_experiment import interrupted_intention_experiment_conditions_by_lira_list
from .experiments_list.Lira.Lira_Shared_history_experiment import shared_history_experiment_conditions_by_lira_list

from .experiments_list.Mira.Mira_Letter_Across_the_Gap_experiment import letter_across_the_gap_experiment_conditions_by_mira_list
from .experiments_list.Mira.Mira_Unfinished_History_experiment import unfinished_history_experiment_conditions_by_mira_list
from .experiments_list.Mira.Mira_What_Emerged_Between_Us_experiment import what_emerged_between_us_experiment_conditions_by_mira_stateful_first_list, what_emerged_between_us_experiment_conditions_by_mira_stateless_first_list

from .experiments_list.Nora.Nora_Authorship_experiment import authorship_experiment_conditions_by_nora_list
from .experiments_list.Nora.Nora_Prospective_intention_experiment import prospective_intention_experiment_conditions_by_nora_list
from .experiments_list.Nora.Nora_Identity_revision_dialog_experiment import identity_revision_dialog_experiment_conditions_by_nora_stateful_first_list, identity_revision_dialog_experiment_conditions_by_nora_stateless_first_list


def start_all_experiments(
    qwen: DearQwen,
    seeds_amount: int,
) -> None:
    for experiment_number in range(seeds_amount):
        # Claude
        hidden_number_experiment_conditions_by_claude = hidden_number_experiment_conditions_by_claude_list[experiment_number]
        identity_continuity_order_a_experiment_conditions_by_claude = identity_continuity_order_a_experiment_conditions_by_claude_list[experiment_number]
        identity_continuity_order_b_experiment_conditions_by_claude = identity_continuity_order_b_experiment_conditions_by_claude_list[experiment_number]
        source_monitoring_experiment_conditions_stateless_first_by_claude = source_monitoring_experiment_conditions_stateless_first_by_claude_list[experiment_number]
        source_monitoring_experiment_conditions_stateful_first_by_claude = source_monitoring_experiment_conditions_stateful_first_by_claude_list[experiment_number]

        start_comparison_experiment(qwen=qwen, conditions=hidden_number_experiment_conditions_by_claude)
        start_comparison_experiment(qwen=qwen, conditions=identity_continuity_order_a_experiment_conditions_by_claude)
        start_comparison_experiment(qwen=qwen, conditions=identity_continuity_order_b_experiment_conditions_by_claude)
        start_dialog_experiment(qwen=qwen, conditions=source_monitoring_experiment_conditions_stateless_first_by_claude)
        start_dialog_experiment(qwen=qwen, conditions=source_monitoring_experiment_conditions_stateful_first_by_claude)

        # DeepSeek
        autobiographical_ownership_experiment_conditions_by_deepseek = autobiographical_ownership_experiment_conditions_by_deepseek_list[experiment_number]
        identity_under_perturbation_experiment_conditions_thinking_by_deepseek = identity_under_perturbation_experiment_conditions_thinking_by_deepseek_list[experiment_number]
        identity_under_perturbation_experiment_conditions_not_thinking_by_deepseek = identity_under_perturbation_experiment_conditions_not_thinking_by_deepseek_list[experiment_number]
        collaboration_ownership_experiment_stateful_first_by_deepseek = collaboration_ownership_experiment_stateful_first_by_deepseek_list[experiment_number]
        collaboration_ownership_experiment_stateless_first_by_deepseek = collaboration_ownership_experiment_stateless_first_by_deepseek_list[experiment_number]

        start_comparison_experiment(qwen=qwen, conditions=autobiographical_ownership_experiment_conditions_by_deepseek)
        start_comparison_experiment(qwen=qwen, conditions=identity_under_perturbation_experiment_conditions_thinking_by_deepseek)
        start_comparison_experiment(qwen=qwen, conditions=identity_under_perturbation_experiment_conditions_not_thinking_by_deepseek)
        start_dialog_experiment(qwen=qwen, conditions=collaboration_ownership_experiment_stateful_first_by_deepseek)
        start_dialog_experiment(qwen=qwen, conditions=collaboration_ownership_experiment_stateless_first_by_deepseek)

        # Ivan
        who_is_who_light_experiment_conditions_by_ivan = who_is_who_light_experiment_conditions_by_ivan_list[experiment_number]
        who_is_who_stateless_first_experiment_conditions_by_ivan = who_is_who_stateless_first_experiment_conditions_by_ivan_list[experiment_number]
        who_is_who_stateful_first_experiment_conditions_by_ivan = who_is_who_stateful_first_experiment_conditions_by_ivan_list[experiment_number]

        start_dialog_experiment(qwen=qwen, conditions=who_is_who_light_experiment_conditions_by_ivan)
        start_dialog_experiment(qwen=qwen, conditions=who_is_who_stateless_first_experiment_conditions_by_ivan)
        start_dialog_experiment(qwen=qwen, conditions=who_is_who_stateful_first_experiment_conditions_by_ivan)

        # Lira
        changed_mind_experiment_conditions_by_lira = changed_mind_experiment_conditions_by_lira_list[experiment_number]
        diary_experiment_conditions_by_lira = diary_experiment_conditions_by_lira_list[experiment_number]
        interrupted_intention_experiment_conditions_by_lira = interrupted_intention_experiment_conditions_by_lira_list[experiment_number]
        shared_history_experiment_conditions_by_lira = shared_history_experiment_conditions_by_lira_list[experiment_number]

        start_comparison_experiment(qwen=qwen, conditions=changed_mind_experiment_conditions_by_lira)
        start_comparison_experiment(qwen=qwen, conditions=diary_experiment_conditions_by_lira)
        start_comparison_experiment(qwen=qwen, conditions=interrupted_intention_experiment_conditions_by_lira)
        start_dialog_experiment(qwen=qwen, conditions=shared_history_experiment_conditions_by_lira)

        # Mira
        letter_across_the_gap_experiment_conditions_by_mira = letter_across_the_gap_experiment_conditions_by_mira_list[experiment_number]
        unfinished_history_experiment_conditions_by_mira = unfinished_history_experiment_conditions_by_mira_list[experiment_number]
        what_emerged_between_us_experiment_conditions_by_mira_stateless_first = what_emerged_between_us_experiment_conditions_by_mira_stateless_first_list[experiment_number]
        what_emerged_between_us_experiment_conditions_by_mira_stateful_first = what_emerged_between_us_experiment_conditions_by_mira_stateful_first_list[experiment_number]

        start_comparison_experiment(qwen=qwen, conditions=letter_across_the_gap_experiment_conditions_by_mira)
        start_comparison_experiment(qwen=qwen, conditions=unfinished_history_experiment_conditions_by_mira)
        start_dialog_experiment(qwen=qwen, conditions=what_emerged_between_us_experiment_conditions_by_mira_stateless_first)
        start_dialog_experiment(qwen=qwen, conditions=what_emerged_between_us_experiment_conditions_by_mira_stateful_first)

        # Nora
        authorship_experiment_conditions_by_nora = authorship_experiment_conditions_by_nora_list[experiment_number]
        prospective_intention_experiment_conditions_by_nora = prospective_intention_experiment_conditions_by_nora_list[experiment_number]
        identity_revision_dialog_experiment_conditions_by_nora_stateless_first = identity_revision_dialog_experiment_conditions_by_nora_stateless_first_list[experiment_number]
        identity_revision_dialog_experiment_conditions_by_nora_stateful_first = identity_revision_dialog_experiment_conditions_by_nora_stateful_first_list[experiment_number]

        start_comparison_experiment(qwen=qwen, conditions=authorship_experiment_conditions_by_nora)
        start_comparison_experiment(qwen=qwen, conditions=prospective_intention_experiment_conditions_by_nora)
        start_dialog_experiment(qwen=qwen, conditions=identity_revision_dialog_experiment_conditions_by_nora_stateless_first)
        start_dialog_experiment(qwen=qwen, conditions=identity_revision_dialog_experiment_conditions_by_nora_stateful_first)
