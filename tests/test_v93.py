from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
import pytest
from tsaime.calendar_statistics import calendar_frame, calendar_skill_intervals, complete_week_shift_diagnostic

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ns():
    notebook=nbformat.read(ROOT/'notebooks/tsAIME_APR_all_experiments_v9_3.ipynb',as_version=4)
    scope={}
    # Definition cells only; never installation, downloads, or full experiments.
    for index in (5,7,9,11,13,15):
        exec(compile(notebook.cells[index].source,f'v93-cell-{index}','exec'),scope)
    return scope


def test_calendar_ci_keeps_gaps_and_exact_constant_skill():
    dates=pd.date_range('2024-01-01',periods=400,freq='6h').delete([10,11,12,99])
    n=len(dates)
    table=calendar_skill_intervals(dates,np.zeros(n),np.ones(n),np.ones(n)*2,
        interval_hours=6,n_resamples=99)
    np.testing.assert_allclose(table[['skill','CI_low','CI_high']],.5)
    assert set(table.grid_n)=={400}
    assert set(table.n)=={396}
    assert set(table.block_days)=={7,14,28}


def test_calendar_validation():
    with pytest.raises(ValueError,match='unique'):
        calendar_frame(pd.to_datetime(['2024-01-01','2024-01-01']),np.zeros(2),6)
    with pytest.raises(ValueError,match='grid'):
        calendar_frame(pd.DatetimeIndex([pd.Timestamp('2024-01-01'),pd.Timestamp('2024-01-01 07:00')]),np.zeros(2),6)


def test_full_week_edges_on_nonmultiple_and_missing_dates():
    dates=pd.date_range('2024-01-03 06:00',periods=28*12+23,freq='6h')
    keep=np.ones(len(dates),bool);keep[[20,30,50]]=False;dates=dates[keep]
    rng=np.random.default_rng(7);x=rng.normal(size=(len(dates),4));y=x[:,:2]+rng.normal(size=(len(dates),2))*.1
    summary,detail=complete_week_shift_diagnostic(dates,x,y,ridge=.01,interval_hours=6)
    r=summary.iloc[0]
    assert r.phase_mismatch_count==0
    assert r.grid_start.weekday()==0 and r.grid_start.hour==0
    assert (r.grid_end_exclusive-r.grid_start).days%7==0
    assert len(detail)==r.full_weeks-1
    assert detail.shift_weeks.is_unique
    assert detail.paired_n.min()>0
    assert r.discarded_grid_rows>0


def test_unit_invariant_regularization_selection(ns):
    rng=np.random.default_rng(23);x=rng.normal(size=(150,10));y=x[:,:3]+rng.normal(size=(150,3))*.5
    scale=np.array([100,1,1,1,1,100,1,1,1,1])
    a,t1=ns['_select_inverse_ridge'](x,y,(0,.001,.01,.1,1))
    b,t2=ns['_select_inverse_ridge'](x*scale,y,(0,.001,.01,.1,1))
    assert a==b
    np.testing.assert_allclose(t1.inverse_RMSE,t2.inverse_RMSE,rtol=1e-10)


def test_quadratic_uses_mean_loss_lambda_and_is_unit_invariant(ns):
    rng=np.random.default_rng(5);y=rng.normal(size=(150,3));x=rng.normal(size=(150,10));x[:,:3]+=y**2
    out,t1=ns['_quadratic_inverse'](x,y,y[:12],(0,.01,.1,1))
    factor=np.arange(1,11)*100
    out2,t2=ns['_quadratic_inverse'](x*factor,y,y[:12],(0,.01,.1,1))
    np.testing.assert_allclose(t1.sklearn_alpha,t1.fit_n*t1.ridge)
    np.testing.assert_allclose(t1.validation_inverse_RMSE,t2.validation_inverse_RMSE,rtol=1e-10)
    np.testing.assert_allclose(out,out2/factor,rtol=1e-9,atol=1e-10)


def test_duplicate_output_marginal_average_control(ns):
    rng=np.random.default_rng(11);x=rng.normal(size=(200,10));y=np.repeat(x[:,[0]],3,axis=1)
    pred,_=ns['_inverse_predictions'](x,y,y,0,np.zeros(10,dtype=int),np.zeros(3))
    np.testing.assert_allclose(pred['linear_tsAIME'][:,0],x[:,0],atol=1e-12)
    np.testing.assert_allclose(pred['marginal_mean'][:,0],x[:,0],atol=1e-12)
    assert np.mean((pred['marginal_sum_ablation'][:,0]-x[:,0])**2)>1


def test_rolling_calibration_strictly_precedes_evaluation(ns):
    rng=np.random.default_rng(7);x=rng.normal(size=(250,10));y=x[:,:3]
    cfg=ns['APRExperimentConfig'](run_mode='smoke').resolved()
    dates=pd.date_range('2024-01-01',periods=len(x),freq='24h')
    ds,obs,pred,windows=ns['_rolling_evaluation'](dates,x,y,.01,np.zeros(10,dtype=int),np.zeros(3),cfg)
    assert ds.is_unique
    assert all(r['calibration_end']<r['evaluation_start'] for r in windows)
    assert set(('past_mean','marginal_mean','validation_selected_single')).issubset(pred)
    assert len(obs)==len(ds)


def test_serial_synthetic_has_real_transitions_and_oos(ns):
    cfg=ns['APRExperimentConfig'](run_mode='smoke',synthetic_regimes=1,
        synthetic_rows_per_regime=100,synthetic_tracking_replicates=1).resolved()
    t=ns['_synthetic_v93'](cfg)
    assert set(t['synthetic_tracking'].regime)=={0,1,2,3}
    assert t['synthetic_tracking'].mixed_calibration_regimes.any()
    assert (t['synthetic_tracking'].calibration_end_index<t['synthetic_tracking'].evaluation_start_index).all()
    assert t['synthetic_recovery'].normal_equation_error.max()<1e-10
    assert t['scalar'].absolute_difference.max()<1e-10


def test_historical_v93_notebook_compiles_and_self_contains_apr():
    n=nbformat.read(ROOT/'notebooks/tsAIME_APR_all_experiments_v9_3.ipynb',as_version=4)
    for i,c in enumerate(n.cells):
        if c.cell_type=='code':
            compile(c.source,f'cell-{i}','exec')
            # Preserve the user's execution record. Release cleanliness is
            # tested against the new v9.4 notebook, not this historical file.
    source='\n'.join(c.source for c in n.cells)
    assert '/private/tmp/' not in source and '/Users/' not in source
    assert 'from experiments' not in source
    assert 'def _run_site_v93' in source and 'def run_apr_pm25' in source


def test_legacy_shift_discards_incomplete_final_period():
    from tsaime import structured_period_shift_test
    rng=np.random.default_rng(19)
    dates=pd.date_range('2024-01-01',periods=28*6+23,freq='6h')
    x=rng.normal(size=(len(dates),3));y=x[:,:2]+rng.normal(size=(len(dates),2))*.2
    a=structured_period_shift_test(dates,x,y,ridge=.01,sampling_interval_hours=6)
    b=structured_period_shift_test(dates[:28*6],x[:28*6],y[:28*6],ridge=.01,sampling_interval_hours=6)
    np.testing.assert_allclose(a[0],b[0]);np.testing.assert_allclose(a[2],b[2])


def test_fast_calendar_bootstrap_agrees_with_direct_sampling():
    rng=np.random.default_rng(12);dates=pd.date_range('2024-01-01',periods=45,freq='24h')
    obs=rng.normal(size=45);pred=obs+rng.normal(size=45);base=obs+rng.normal(size=45)*2
    obs[[8,9,10]]=np.nan
    result=calendar_skill_intervals(dates,obs,pred,base,interval_hours=24,block_days=(7,),n_resamples=99,seed=45)
    rng=np.random.default_rng(45);indices=[]
    for width in [7]*6+[3]:
        starts=rng.integers(0,45-width+1,99)
        indices.append(starts[:,None]+np.arange(width))
    ix=np.concatenate(indices,axis=1)
    a=np.nansum((pred[ix]-obs[ix])**2,axis=1);b=np.nansum((base[ix]-obs[ix])**2,axis=1)
    expected=np.quantile(1-np.sqrt(a/b),[.025,.975])
    np.testing.assert_allclose(result[['CI_low','CI_high']].iloc[0],expected)


def test_forecast_latex_summary_does_not_pool_other_models(ns):
    frame=pd.DataFrame(dict(dataset=['A','A'],site=['S','S'],primary_interval=[True,True],
        model=['SMap','training_mean'],baseline=['persistence','persistence'],horizon_h=[6,6],
        RMSE=[10.,100.],skill=[.2,-7.],CI_low=[.1,-8.]))
    summary=ns['_compact_v93']('table08_forecast',frame)
    assert summary.iloc[0].median_RMSE==10.
    assert summary.iloc[0].median_skill==.2
    assert summary.iloc[0].positive_pointwise_intervals==1


def test_cross_task_summary_keeps_distinct_estimands(ns):
    frame=pd.DataFrame(dict(task=['forward','inverse'],method=['F','A'],RMSE=[.1,1.]))
    summary=ns['_compact_v93']('table03_synthetic_cross_task',frame)
    assert len(summary)==2
    np.testing.assert_allclose(summary.median_RMSE,[.1,1.])
