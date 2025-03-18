import base64, io, pandas as pd
import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.graph_objs as go
from datetime import datetime

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Shrinkage Control Charts (Expected - Measured)"),
    
    # File Upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
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
    
    # Filter panel (hidden until file is uploaded)
    html.Div(id='filter-panel', style={'display': 'none'}, children=[
        html.Label("Select Material:"),
        dcc.Dropdown(id='material-dropdown'),
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-picker',
            start_date_placeholder_text="Start Period",
            end_date_placeholder_text="End Period"
        ),
        html.H3("Set Control and Specification Limits (for Deviation)"),
        html.Div([
            html.Div([
                html.Label("Width Control Delta:"),
                dcc.Input(id='width-control-delta', type='number', value=1, step=0.1)
            ], style={'display': 'inline-block', 'margin-right': '20px'}),
            html.Div([
                html.Label("Width Spec Delta:"),
                dcc.Input(id='width-spec-delta', type='number', value=1.5, step=0.1)
            ], style={'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([
                html.Label("Length Control Delta:"),
                dcc.Input(id='length-control-delta', type='number', value=1, step=0.1)
            ], style={'display': 'inline-block', 'margin-right': '20px'}),
            html.Div([
                html.Label("Length Spec Delta:"),
                dcc.Input(id='length-spec-delta', type='number', value=1.5, step=0.1)
            ], style={'display': 'inline-block'})
        ])
    ]),
    
    # Summary stats
    html.Div(id='summary-stats'),
    
    # Charts container
    html.Div(id='charts-container', style={'display': 'none'}, children=[
        # Row for Width
        html.Div([
            html.Div([
                html.H3("Width Shrinkage Deviation Control Chart"),
                dcc.Graph(id='width-control-chart')
            ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.H3("Width Deviation Distribution"),
                dcc.Graph(id='width-distribution-chart')
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
        ]),
        # Row for Length
        html.Div([
            html.Div([
                html.H3("Length Shrinkage Deviation Control Chart"),
                dcc.Graph(id='length-control-chart')
            ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.H3("Length Deviation Distribution"),
                dcc.Graph(id='length-distribution-chart')
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
        ])
    ]),
    
    # Drill-down details container
    html.Div(id='drilldown-record', style={
        'whiteSpace': 'pre-wrap',
        'border': '1px solid #ccc',
        'padding': '10px',
        'margin': '10px'
    })
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Convert percentage strings to floats for relevant columns
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

def generate_deviation_chart(df, expected_col, measured_col, title, control_delta, spec_delta):
    # Sort by time
    df = df.sort_values('_dry_ts')
    # Compute Deviation = Expected - Measured
    df['Deviation'] = df[expected_col] - df[measured_col]
    
    # Horizontal lines: 0, ± control_delta, ± spec_delta
    trace_measured = go.Scatter(
        x=df['_dry_ts'],
        y=df['Deviation'],
        mode='markers',
        name='Deviation (Expected - Measured)'
    )
    trace_center = go.Scatter(
        x=df['_dry_ts'],
        y=[0]*len(df),
        mode='lines',
        name='Center (0)'
    )
    trace_upper_control = go.Scatter(
        x=df['_dry_ts'],
        y=[control_delta]*len(df),
        mode='lines',
        line=dict(dash='dash'),
        name='Upper Control Limit'
    )
    trace_lower_control = go.Scatter(
        x=df['_dry_ts'],
        y=[-control_delta]*len(df),
        mode='lines',
        line=dict(dash='dash'),
        name='Lower Control Limit'
    )
    trace_upper_spec = go.Scatter(
        x=df['_dry_ts'],
        y=[spec_delta]*len(df),
        mode='lines',
        line=dict(dash='dot'),
        name='Upper Spec Limit'
    )
    trace_lower_spec = go.Scatter(
        x=df['_dry_ts'],
        y=[-spec_delta]*len(df),
        mode='lines',
        line=dict(dash='dot'),
        name='Lower Spec Limit'
    )

    layout = go.Layout(
        title=title,
        hovermode='closest',
        xaxis={'title': 'Measurement Date'},
        yaxis={'title': f"{title} Deviation (Expected - Measured)"}
    )
    fig = go.Figure(
        data=[
            trace_measured, trace_center,
            trace_upper_control, trace_lower_control,
            trace_upper_spec, trace_lower_spec
        ],
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
    Input('length-spec-delta', 'value')
)
def update_charts(data, selected_material, start_date, end_date,
                  width_control_delta, width_spec_delta,
                  length_control_delta, length_spec_delta):
    if data is None or not selected_material:
        return {'display': 'none'}, {}, {}, {}, {}, ""
    
    df = pd.read_json(data, orient='split')
    df['_dry_ts'] = pd.to_datetime(df['_dry_ts'])
    mask = ((df["Approved Materials"] == selected_material) & 
            (df['_dry_ts'] >= start_date) & 
            (df['_dry_ts'] <= end_date))
    filtered = df.loc[mask]
    
    if filtered.empty:
        return {'display': 'block'}, {}, {}, {}, {}, html.Div("No data for selected filters.")
    
    # Generate control & distribution charts for Width
    width_fig = generate_deviation_chart(
        filtered, "Estimated Shrinkage X", "Expected Width Shrinkage (Roll Inspection)",
        "Width Shrinkage", width_control_delta, width_spec_delta
    )
    width_dist_fig = generate_deviation_distribution_chart(
        filtered, "Estimated Shrinkage X", "Expected Width Shrinkage (Roll Inspection)",
        "Width Deviation Distribution"
    )
    
    # Generate control & distribution charts for Length
    length_fig = generate_deviation_chart(
        filtered, "Estimated Shrinkage Y", "Expected Length Shrinkage (Roll Inspection)",
        "Length Shrinkage", length_control_delta, length_spec_delta
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
            f"Width Deviation: {out_control_w} out of {total_w} points out of control ({pct_control_w}), "
            f"{out_spec_w} out of spec ({pct_spec_w})."
        ),
        html.P(
            f"Length Deviation: {out_control_l} out of {total_l} points out of control ({pct_control_l}), "
            f"{out_spec_l} out of spec ({pct_spec_l})."
        )
    ])
    
    return {'display': 'block'}, width_fig, width_dist_fig, length_fig, length_dist_fig, summary

# Drill-down callback: when a point is clicked, display all record data as a formatted list.
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
    
    # Determine which chart triggered the callback.
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
    
    # Extract clicked timestamp and deviation value.
    clicked_time = pd.to_datetime(clickData['points'][0]['x'])
    clicked_deviation = clickData['points'][0]['y']  # Deviation = Expected - Measured
    df = pd.read_json(data, orient='split')
    df['_dry_ts'] = pd.to_datetime(df['_dry_ts'])
    # Recalculate deviation for the relevant columns.
    df['Deviation'] = df[expected_col] - df[measured_col]
    
    # Define a tolerance for matching the clicked time.
    tolerance = pd.Timedelta(seconds=1)
    matching_rows = df[
        (df['_dry_ts'] >= (clicked_time - tolerance)) &
        (df['_dry_ts'] <= (clicked_time + tolerance)) &
        (df['Deviation'].sub(clicked_deviation).abs() < 1e-6)
    ]
    
    if matching_rows.empty:
        return "No matching record found."
    
    # For this example, we display the first matching record.
    record = matching_rows.iloc[0]
    details = []
    for col in record.index:
        details.append(html.Li(f"{col}: {record[col]}"))
    
    return html.Div([
        html.H4(f"{chart_name} Drill Down Record"),
        html.Ul(details)
    ], style={'border': '1px solid #ccc', 'padding': '10px', 'margin': '10px'})

if __name__ == '__main__':
    app.run_server(debug=True)
