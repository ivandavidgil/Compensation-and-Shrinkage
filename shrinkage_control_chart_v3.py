import base64, io, pandas as pd
import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.graph_objs as go
from datetime import datetime

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Shrinkage Control Charts (Expected - Measured)"),
    
    # File Upload component.
    dcc.Upload(
        id='upload-data',
        children=html.Div(['UPLOAD ', html.A('DATA')]),
        style={
            'width': '98%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='upload-status', children="No data uploaded yet."),
    
    dcc.Store(id='stored-data'),
    
    # Filter panel (hidden until file is uploaded).
    html.Div(id='filter-panel', style={'display': 'none'}, children=[
        html.Label("Select Material:"),
        dcc.Dropdown(id='material-dropdown'),
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-picker',
            start_date_placeholder_text="Start Period",
            end_date_placeholder_text="End Period"
        ),
        html.H3("Set Specification Limits (for Deviation)"),
        html.Div([
            html.Div([
                html.Label("Width Spec 1 Delta:"),
                dcc.Input(id='width-control-delta', type='number', value=1, step=0.1)
            ], style={'display': 'inline-block', 'margin-right': '20px'}),
            html.Div([
                html.Label("Width Spec 2 Delta:"),
                dcc.Input(id='width-spec-delta', type='number', value=1.5, step=0.1)
            ], style={'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([
                html.Label("Length Spec 1 Delta:"),
                dcc.Input(id='length-control-delta', type='number', value=1, step=0.1)
            ], style={'display': 'inline-block', 'margin-right': '20px'}),
            html.Div([
                html.Label("Length Spec 2 Delta:"),
                dcc.Input(id='length-spec-delta', type='number', value=1.5, step=0.1)
            ], style={'display': 'inline-block'})
        ]),
        # New Median Toggle Control:
        html.Div([
            html.Label("Display Median in Control Charts:"),
            dcc.Checklist(
                id='show-median-toggle',
                options=[{'label': 'Show Median', 'value': 'show'}],
                value=[]  # Off by default
            )
        ], style={'margin-top': '20px'})
    ]),
    
    # Summary statistics.
    html.Div(id='summary-stats'),
    
    # Charts container.
    html.Div(id='charts-container', style={'display': 'none'}, children=[
        # Row for Width charts.
        html.Div([
            html.Div([
                html.H3("Width Shrinkage Deviation Control Chart"),
                dcc.Graph(id='width-control-chart'),
                html.Button("Expand", id="expand-width-control", n_clicks=0)
            ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.H3("Width Deviation Distribution"),
                dcc.Graph(id='width-distribution-chart'),
                html.Button("Expand", id="expand-width-dist", n_clicks=0)
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
        ]),
        # Row for Length charts.
        html.Div([
            html.Div([
                html.H3("Length Shrinkage Deviation Control Chart"),
                dcc.Graph(id='length-control-chart'),
                html.Button("Expand", id="expand-length-control", n_clicks=0)
            ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.H3("Length Deviation Distribution"),
                dcc.Graph(id='length-distribution-chart'),
                html.Button("Expand", id="expand-length-dist", n_clicks=0)
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
        ])
    ]),
    
    # Drill-down details container.
    html.Div(id='drilldown-record', style={
        'whiteSpace': 'pre-wrap',
        'border': '1px solid #ccc',
        'padding': '10px',
        'margin': '10px'
    }),
    
    # Full-screen modal for expanded charts.
    html.Div(
        id='fullscreen-modal',
        style={
            'display': 'none',
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'width': '100vw',
            'height': '100vh',
            'backgroundColor': 'rgba(0, 0, 0, 0.8)',
            'zIndex': 1000,
            'padding': '20px'
        },
        children=[
            html.Button(
                "Close",
                id="close-modal",
                n_clicks=0,
                style={
                    'position': 'absolute',
                    'top': '40px',
                    'right': '50px',
                    'zIndex': 1100,
                    'fontSize': '20px',
                    'padding': '10px 20px'
                }
            ),
            dcc.Graph(id='fullscreen-graph', style={'height': '90vh'})
        ]
    )
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Convert percentage strings to floats for relevant columns.
            percentage_columns = [
                "Expected Width Shrinkage (Roll Inspection)",
                "Expected Length Shrinkage (Roll Inspection)",
                "Estimated Shrinkage X",
                "Estimated Shrinkage Y"
            ]
            for col in percentage_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.rstrip('%').astype(float)
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df

@app.callback(
    Output('stored-data', 'data'),
    Output('upload-status', 'children'),
    Output('filter-panel', 'style'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            df['_dry_ts'] = pd.to_datetime(df['_dry_ts'])
            return df.to_json(date_format='iso', orient='split'), f"Data uploaded: {filename}", {'display': 'block'}
    return dash.no_update, "No data uploaded yet.", {'display': 'none'}

@app.callback(
    Output('material-dropdown', 'options'),
    Output('material-dropdown', 'value'),
    Output('date-picker', 'min_date_allowed'),
    Output('date-picker', 'max_date_allowed'),
    Output('date-picker', 'start_date'),
    Output('date-picker', 'end_date'),
    Input('stored-data', 'data')
)
def update_filters(data):
    if data is None:
        return [], None, None, None, None, None
    df = pd.read_json(data, orient='split')
    df['_dry_ts'] = pd.to_datetime(df['_dry_ts'])
    materials = df["Approved Materials"].dropna().unique()
    options = [{'label': mat, 'value': mat} for mat in materials]
    min_date = df['_dry_ts'].min().date()
    max_date = df['_dry_ts'].max().date()
    return options, (options[0]['value'] if options else None), min_date, max_date, min_date, max_date

def generate_deviation_chart(df, expected_col, measured_col, title, control_delta, spec_delta, show_median=False):
    df = df.sort_values('_dry_ts')
    df['Deviation'] = df[expected_col] - df[measured_col]
    
    trace_measured = go.Scatter(
        x=df['_dry_ts'],
        y=df['Deviation'],
        mode='markers',
        name='Deviation (Expected - Measured)'
    )
    trace_center = go.Scatter(
        x=df['_dry_ts'],
        y=[0] * len(df),
        mode='lines',
        name='Center (0)'
    )
    trace_upper_control = go.Scatter(
        x=df['_dry_ts'],
        y=[control_delta] * len(df),
        mode='lines',
        line=dict(dash='dash'),
        name='Upper Spec 1 Limit'
    )
    trace_lower_control = go.Scatter(
        x=df['_dry_ts'],
        y=[-control_delta] * len(df),
        mode='lines',
        line=dict(dash='dash'),
        name='Lower Spec 1 Limit'
    )
    trace_upper_spec = go.Scatter(
        x=df['_dry_ts'],
        y=[spec_delta] * len(df),
        mode='lines',
        line=dict(dash='dot'),
        name='Upper Spec 2 Limit'
    )
    trace_lower_spec = go.Scatter(
        x=df['_dry_ts'],
        y=[-spec_delta] * len(df),
        mode='lines',
        line=dict(dash='dot'),
        name='Lower Spec 2 Limit'
    )
    
    traces = [trace_measured, trace_center, trace_upper_control, trace_lower_control, trace_upper_spec, trace_lower_spec]
    
    # Add the median line if toggled on.
    if show_median:
        median_val = df['Deviation'].median()
        trace_median = go.Scatter(
            x=df['_dry_ts'],
            y=[median_val] * len(df),
            mode='lines',
            line=dict(color='purple', dash='dashdot'),
            name=f"Median: {median_val:.2f}"
        )
        traces.append(trace_median)
    
    layout = go.Layout(
        title=title,
        hovermode='closest',
        xaxis={'title': 'Measurement Date'},
        yaxis={'title': f"{title} Deviation (Expected - Measured)"}
    )
    fig = go.Figure(
        data=traces,
        layout=layout
    )
    return fig

def generate_deviation_distribution_chart(df, expected_col, measured_col, title):
    df = df.sort_values('_dry_ts')
    df['Deviation'] = df[expected_col] - df[measured_col]
    trace = go.Histogram(
        y=df['Deviation'],
        orientation='h',
        name='Deviation Distribution',
        opacity=0.75
    )
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Count'},
        yaxis={'title': 'Deviation (Expected - Measured)'},
        bargap=0.2
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig

@app.callback(
    Output('charts-container', 'style'),
    Output('width-control-chart', 'figure'),
    Output('width-distribution-chart', 'figure'),
    Output('length-control-chart', 'figure'),
    Output('length-distribution-chart', 'figure'),
    Output('summary-stats', 'children'),
    Input('stored-data', 'data'),
    Input('material-dropdown', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('width-control-delta', 'value'),
    Input('width-spec-delta', 'value'),
    Input('length-control-delta', 'value'),
    Input('length-spec-delta', 'value'),
    Input('show-median-toggle', 'value')  # New input for median toggle
)
def update_charts(data, selected_material, start_date, end_date,
                  width_control_delta, width_spec_delta,
                  length_control_delta, length_spec_delta,
                  median_toggle_value):
    if data is None or not selected_material:
        return {'display': 'none'}, {}, {}, {}, {}, ""
    
    df = pd.read_json(data, orient='split')
    df['_dry_ts'] = pd.to_datetime(df['_dry_ts'])
    # Adjust filter to include the full end date.
    mask = (
        (df["Approved Materials"] == selected_material) &
        (df['_dry_ts'] >= pd.to_datetime(start_date)) &
        (df['_dry_ts'] < pd.to_datetime(end_date) + pd.Timedelta(days=1))
    )
    filtered = df.loc[mask]
    
    if filtered.empty:
        return {'display': 'block'}, {}, {}, {}, {}, html.Div("No data for selected filters.")
    
    # Determine whether the median should be shown.
    show_median = 'show' in median_toggle_value
    
    width_fig = generate_deviation_chart(
        filtered, "Estimated Shrinkage X", "Expected Width Shrinkage (Roll Inspection)",
        "Width Shrinkage", width_control_delta, width_spec_delta, show_median
    )
    width_dist_fig = generate_deviation_distribution_chart(
        filtered, "Estimated Shrinkage X", "Expected Width Shrinkage (Roll Inspection)",
        "Width Deviation Distribution"
    )
    length_fig = generate_deviation_chart(
        filtered, "Estimated Shrinkage Y", "Expected Length Shrinkage (Roll Inspection)",
        "Length Shrinkage", length_control_delta, length_spec_delta, show_median
    )
    length_dist_fig = generate_deviation_distribution_chart(
        filtered, "Estimated Shrinkage Y", "Expected Length Shrinkage (Roll Inspection)",
        "Length Deviation Distribution"
    )
    
    def calc_summary(df, expected_col, measured_col, cdelta, sdelta):
        df['Deviation'] = df[expected_col] - df[measured_col]
        dev = df['Deviation']
        out_of_control = ((dev > cdelta) | (dev < -cdelta)).sum()
        out_of_spec = ((dev > sdelta) | (dev < -sdelta)).sum()
        total = len(df)
        pct_control = f"{(out_of_control/total*100):.1f}%" if total > 0 else "N/A"
        pct_spec = f"{(out_of_spec/total*100):.1f}%" if total > 0 else "N/A"
        return out_of_control, pct_control, out_of_spec, pct_spec, total

    out_control_w, pct_control_w, out_spec_w, pct_spec_w, total_w = calc_summary(
        filtered, "Estimated Shrinkage X", "Expected Width Shrinkage (Roll Inspection)",
        width_control_delta, width_spec_delta
    )
    out_control_l, pct_control_l, out_spec_l, pct_spec_l, total_l = calc_summary(
        filtered, "Estimated Shrinkage Y", "Expected Length Shrinkage (Roll Inspection)",
        length_control_delta, length_spec_delta
    )
    
    summary = html.Div([
        html.H4("Summary Statistics (Deviation)"),
        html.P(
            f"Width Deviation: {out_control_w} out of {total_w} points out of spec 1 ({pct_control_w}), "
            f"{out_spec_w} out of spec 2 ({pct_spec_w})."
        ),
        html.P(
            f"Length Deviation: {out_control_l} out of {total_l} points out of spec 1 ({pct_control_l}), "
            f"{out_spec_l} out of spec 2 ({pct_spec_l})."
        )
    ])
    
    return {'display': 'block'}, width_fig, width_dist_fig, length_fig, length_dist_fig, summary

@app.callback(
    Output('drilldown-record', 'children'),
    [Input('width-control-chart', 'clickData'),
     Input('length-control-chart', 'clickData')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def display_drilldown(clickData_width, clickData_length, data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == "width-control-chart":
        expected_col = "Estimated Shrinkage X"
        measured_col = "Expected Width Shrinkage (Roll Inspection)"
        chart_name = "Width"
        clickData = clickData_width
    elif triggered_id == "length-control-chart":
        expected_col = "Estimated Shrinkage Y"
        measured_col = "Expected Length Shrinkage (Roll Inspection)"
        chart_name = "Length"
        clickData = clickData_length
    else:
        return dash.no_update
    
    clicked_time = pd.to_datetime(clickData['points'][0]['x'])
    clicked_deviation = clickData['points'][0]['y']  # Deviation = Expected - Measured
    df = pd.read_json(data, orient='split')
    df['_dry_ts'] = pd.to_datetime(df['_dry_ts'])
    df['Deviation'] = df[expected_col] - df[measured_col]
    
    tolerance = pd.Timedelta(seconds=1)
    matching_rows = df[
        (df['_dry_ts'] >= (clicked_time - tolerance)) &
        (df['_dry_ts'] <= (clicked_time + tolerance)) &
        (df['Deviation'].sub(clicked_deviation).abs() < 1e-6)
    ]
    
    if matching_rows.empty:
        return "No matching record found."
    
    record = matching_rows.iloc[0]
    details = []
    for col in record.index:
        details.append(html.Li(f"{col}: {record[col]}"))
    
    return html.Div([
        html.H4(f"{chart_name} Drill Down Record"),
        html.Ul(details)
    ], style={'border': '1px solid #ccc', 'padding': '10px', 'margin': '10px'})

# Combined callback to handle expanding charts and closing the modal.
@app.callback(
    Output('fullscreen-graph', 'figure'),
    Output('fullscreen-modal', 'style'),
    [Input('expand-width-control', 'n_clicks'),
     Input('expand-width-dist', 'n_clicks'),
     Input('expand-length-control', 'n_clicks'),
     Input('expand-length-dist', 'n_clicks'),
     Input('close-modal', 'n_clicks')],
    [State('stored-data', 'data'),
     State('material-dropdown', 'value'),
     State('date-picker', 'start_date'),
     State('date-picker', 'end_date'),
     State('width-control-delta', 'value'),
     State('width-spec-delta', 'value'),
     State('length-control-delta', 'value'),
     State('length-spec-delta', 'value'),
     State('show-median-toggle', 'value')]  # New state for median toggle
)
def manage_fullscreen(exp_wc, exp_wd, exp_lc, exp_ld, close_n, data, selected_material,
                      start_date, end_date, width_control_delta, width_spec_delta,
                      length_control_delta, length_spec_delta, median_toggle_value):
    ctx = dash.callback_context
    if not ctx.triggered or data is None:
        return dash.no_update, {'display': 'none'}
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # If the close button was clicked, hide the modal.
    if triggered_id == "close-modal":
        return {}, {'display': 'none'}
    
    df = pd.read_json(data, orient='split')
    df['_dry_ts'] = pd.to_datetime(df['_dry_ts'])
    # Adjust filter to include the full end date.
    mask = (
        (df["Approved Materials"] == selected_material) &
        (df['_dry_ts'] >= pd.to_datetime(start_date)) &
        (df['_dry_ts'] < pd.to_datetime(end_date) + pd.Timedelta(days=1))
    )
    filtered = df.loc[mask]
    if filtered.empty:
        return dash.no_update, {'display': 'none'}
    
    # Determine whether the median should be shown.
    show_median = 'show' in median_toggle_value
    
    if triggered_id == "expand-width-control":
        fig = generate_deviation_chart(
            filtered, "Estimated Shrinkage X", "Expected Width Shrinkage (Roll Inspection)",
            "Width Shrinkage", width_control_delta, width_spec_delta, show_median
        )
    elif triggered_id == "expand-width-dist":
        fig = generate_deviation_distribution_chart(
            filtered, "Estimated Shrinkage X", "Expected Width Shrinkage (Roll Inspection)",
            "Width Deviation Distribution"
        )
    elif triggered_id == "expand-length-control":
        fig = generate_deviation_chart(
            filtered, "Estimated Shrinkage Y", "Expected Length Shrinkage (Roll Inspection)",
            "Length Shrinkage", length_control_delta, length_spec_delta, show_median
        )
    elif triggered_id == "expand-length-dist":
        fig = generate_deviation_distribution_chart(
            filtered, "Estimated Shrinkage Y", "Expected Length Shrinkage (Roll Inspection)",
            "Length Deviation Distribution"
        )
    else:
        fig = {}
    
    modal_style = {
        'display': 'block',
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'width': '100vw',
        'height': '100vh',
        'backgroundColor': 'rgba(0, 0, 0, 0.8)',
        'zIndex': 1000,
        'padding': '20px'
    }
    return fig, modal_style

if __name__ == '__main__':
    app.run_server(debug=True)
