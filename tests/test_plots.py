"""The report figures build from the committed data, and each PNG stays under 500 KB."""
from analysis import plots


def test_setting_labels():
    assert plots.setting_label("{}") == "default"
    assert plots.setting_label('{"lr": 0.0003}') == "learning rate 3e-4"
    assert plots.setting_label('{"ent_coef": 0.0}') == "entropy 0"
    assert plots.setting_label('{"eps_decay_steps": 20000}') == "explore 20k"
    assert plots.setting_label('{"baseline": false}') == "baseline off"


def test_figures_build_and_stay_small(tmp_path):
    plots.set_style()
    out = str(tmp_path)
    made = [plots.fig_learning_curves(out, rule_val=0.1), plots.fig_dqn(out), plots.fig_pg(out),
            plots.fig_convergence(out), plots.fig_hyperparameters(out), plots.fig_generalization(out),
            plots.fig_behaviour(out), plots.fig_paired(out)]
    assert len(made) == 8
    for path in made:
        size = (tmp_path / path.split("/")[-1]).stat().st_size
        assert 10_000 < size < 500_000, (path, size)


def test_sweep_figures_use_the_original_10_settings():
    """PPO settings 10-11 were added after model selection; the convergence and hyperparameter figures leave them out."""
    for algo in plots.ALGOS:
        assert sorted(plots.main_sweep_runs(algo)) == list(range(10))
    assert list(plots.MAIN_SETTINGS) == list(range(10))
