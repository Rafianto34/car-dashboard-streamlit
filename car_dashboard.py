import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="🚗 Interactive Car Sales Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f2f6;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the car data"""
    try:
        df = pd.read_csv('car_prices.csv')
        
        # Data cleaning
        df = df.dropna(subset=['sellingprice', 'odometer', 'year', 'make'])  # Also filter out missing makes
        df = df[df['sellingprice'] > 0]
        df = df[df['odometer'] > 0]
        df = df[df['odometer'] < 500000]  # Remove unrealistic mileage
        df = df[df['sellingprice'] < 100000]  # Remove extreme outliers
        
        # Convert date with proper timezone handling
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)
            df['sale_month'] = df['saledate'].dt.tz_localize(None).dt.to_period('M')
        
        # Clean string columns
        df['make'] = df['make'].astype(str).str.strip()
        df['model'] = df['model'].fillna('Unknown').astype(str).str.strip()
        df['body'] = df['body'].fillna('Unknown')
        df['body_clean'] = df['body'].astype(str).str.title().str.strip()
        
        # Age calculation
        df['age'] = 2015 - df['year']  # Assuming 2015 as reference year
        
        # Price per mile (avoid division by zero)
        df['price_per_mile'] = df['sellingprice'] / df['odometer'].replace(0, 1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_price_mileage_scatter(df, selected_makes=None, price_range=None, year_range=None):
    """Create interactive price vs mileage scatter plot"""
    filtered_df = df.copy()
    
    # Apply filters
    if selected_makes:
        filtered_df = filtered_df[filtered_df['make'].isin(selected_makes)]
    if price_range:
        filtered_df = filtered_df[
            (filtered_df['sellingprice'] >= price_range[0]) & 
            (filtered_df['sellingprice'] <= price_range[1])
        ]
    if year_range:
        filtered_df = filtered_df[
            (filtered_df['year'] >= year_range[0]) & 
            (filtered_df['year'] <= year_range[1])
        ]
    
    # Sample data for performance
    #if len(filtered_df) > 10000:
        #filtered_df = filtered_df.sample(10000)
    
    fig = px.scatter(
        filtered_df,
        x='odometer', 
        y='sellingprice',
        color='make',
        size='year',
        hover_data=['model', 'body_clean', 'condition'],
        title='🎯 Price vs Mileage Analysis',
        labels={'odometer': 'Mileage (miles)', 'sellingprice': 'Selling Price ($)'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_font_size=20,
        title_x=0.5
    )
    
    return fig

def create_time_series(df, selected_makes=None):
    """Create time series of average prices over time"""
    filtered_df = df.copy()
    
    if selected_makes:
        filtered_df = filtered_df[filtered_df['make'].isin(selected_makes)]
    
    # Monthly aggregation with error handling
    try:
        monthly_stats = filtered_df.groupby(['sale_month', 'make']).agg({
            'sellingprice': ['mean', 'count'],
            'odometer': 'mean'
        }).reset_index()
        
        monthly_stats.columns = ['sale_month', 'make', 'avg_price', 'count', 'avg_mileage']
        monthly_stats = monthly_stats[monthly_stats['count'] >= 10]  # Filter for statistical significance
        monthly_stats['sale_date'] = monthly_stats['sale_month'].dt.to_timestamp()
    except Exception as e:
        st.warning(f"Error processing time series data: {e}")
        # Create dummy data for visualization
        monthly_stats = pd.DataFrame({
            'sale_date': pd.date_range('2014-01-01', '2015-12-01', freq='M'),
            'make': ['Ford'] * 24,
            'avg_price': [15000] * 24
        })
    
    fig = px.line(
        monthly_stats,
        x='sale_date',
        y='avg_price',
        color='make',
        title='📈 Average Price Trends Over Time',
        labels={'sale_date': 'Sale Date', 'avg_price': 'Average Price ($)'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=500,
        title_font_size=20,
        title_x=0.5,
        xaxis_title='Sale Date',
        yaxis_title='Average Price ($)'
    )
    
    return fig

def create_brand_comparison(df, metric='sellingprice'):
    """Create brand comparison visualization"""
    top_brands = df['make'].value_counts().head(15).index
    brand_data = df[df['make'].isin(top_brands)]
    
    if metric == 'sellingprice':
        agg_data = brand_data.groupby('make')['sellingprice'].agg(['mean', 'median', 'count']).reset_index()
        fig = px.bar(
            agg_data.sort_values('mean', ascending=True),
            x='mean',
            y='make',
            orientation='h',
            title='💰 Average Price by Brand',
            labels={'mean': 'Average Price ($)', 'make': 'Brand'},
            template='plotly_white',
            color='mean',
            color_continuous_scale='Viridis'
        )
    else:  # Volume
        volume_data = brand_data['make'].value_counts().reset_index()
        volume_data.columns = ['make', 'count']
        fig = px.bar(
            volume_data,
            x='make',
            y='count',
            title='📊 Sales Volume by Brand',
            labels={'count': 'Number of Sales', 'make': 'Brand'},
            template='plotly_white',
            color='count',
            color_continuous_scale='Blues'
        )
        fig.update_xaxes(tickangle=45)
    
    fig.update_layout(height=600, title_font_size=20, title_x=0.5)
    return fig

def create_condition_analysis(df):
    """Create condition vs price analysis"""
    # Clean condition data
    df_clean = df[df['condition'].between(1, 50)]  # Reasonable condition range
    
    fig = px.box(
        df_clean,
        x='condition',
        y='sellingprice',
        title='🔧 Vehicle Condition vs Price Analysis',
        labels={'condition': 'Condition Rating', 'sellingprice': 'Selling Price ($)'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=500,
        title_font_size=20,
        title_x=0.5
    )
    
    return fig

def create_body_type_analysis(df):
    """Create body type market analysis"""
    # Clean and group body types
    body_mapping = {
        'SUV': 'SUV', 'Sedan': 'Sedan', 'Coupe': 'Coupe', 'Convertible': 'Convertible',
        'Wagon': 'Wagon', 'Hatchback': 'Hatchback', 'Minivan': 'Minivan', 'Van': 'Van'
    }
    
    df['body_grouped'] = df['body_clean'].apply(
        lambda x: next((v for k, v in body_mapping.items() if k.lower() in x.lower()), 'Other')
    )
    
    body_stats = df.groupby('body_grouped').agg({
        'sellingprice': ['mean', 'count'],
        'odometer': 'mean'
    }).reset_index()
    
    body_stats.columns = ['body_type', 'avg_price', 'count', 'avg_mileage']
    body_stats = body_stats[body_stats['count'] >= 100]  # Filter for significance
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Price by Body Type', 'Sales Volume by Body Type'),
        specs=[[{"secondary_y": False}, {"type": "pie"}]]
    )
    
    # Bar chart
    fig.add_bar(
        x=body_stats['body_type'],
        y=body_stats['avg_price'],
        name='Avg Price',
        row=1, col=1,
        marker_color='lightblue'
    )
    
    # Pie chart
    fig.add_pie(
        labels=body_stats['body_type'],
        values=body_stats['count'],
        name='Volume',
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        title_text='🚙 Body Type Market Analysis',
        title_font_size=20,
        title_x=0.5,
        showlegend=False
    )
    
    return fig

def create_geographic_distribution(df):
    """Create geographic distribution of sales"""
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['state'] = df_clean['state'].str.upper()  # Convert to uppercase for plotly
    df_clean = df_clean.dropna(subset=['state'])  # Remove missing states
    
    state_stats = df_clean.groupby('state').agg({
        'sellingprice': ['mean', 'count'],
        'odometer': 'mean'
    }).reset_index()
    
    state_stats.columns = ['state', 'avg_price', 'count', 'avg_mileage']
    state_stats = state_stats[state_stats['count'] >= 50]  # Reduced threshold for better coverage
    
    # Debug: Check if we have valid data
    if len(state_stats) == 0:
        st.warning("No state data available for geographic visualization")
        return None
    
    fig = px.choropleth(
        state_stats,
        locations='state',
        color='avg_price',
        hover_data=['count', 'avg_mileage'],
        locationmode='USA-states',
        scope='usa',
        title='🗺️ Average Car Prices by State',
        labels={'avg_price': 'Average Price ($)'},
        color_continuous_scale='Viridis',
        template='plotly_white'
    )
    
    fig.update_layout(
        height=600,
        title_font_size=20,
        title_x=0.5
    )
    
    return fig

def main():
    
    
    
    st.markdown('<h1 class="main-header">🚗 Interactive Car Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    
  
    with st.spinner('Loading car sales data...'):
        df = load_data()
    
    if df.empty:
        st.error("Failed to load data. Please check the car_prices.csv file.")
        return
    
    
    st.sidebar.header("🎛️ Dashboard Controls")
    
   
    available_makes = sorted([make for make in df['make'].unique() if pd.notna(make)])
    selected_makes = st.sidebar.multiselect(
        "Select Car Brands:",
        available_makes,
        default=available_makes[:5],
        help="Choose which car brands to include in the analysis"
    )
   
    min_price, max_price = int(df['sellingprice'].min()), int(df['sellingprice'].max())
    price_range = st.sidebar.slider(
        "Price Range ($):",
        min_value=min_price,
        max_value=min_price + 50000,  # Limit for better UX
        value=(min_price, 30000),
        step=1000,
        help="Filter cars by price range"
    )
    
    
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    year_range = st.sidebar.slider(
        "Model Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(2010, max_year),
        help="Filter cars by model year"
    )
    
   
    analysis_type = st.sidebar.selectbox(
        "Analysis Focus:",
        ["Price Analysis", "Market Trends", "Brand Comparison", "Geographic Distribution"],
        help="Choose the type of analysis to display"
    )
    
    
    filtered_df = df.copy()
    if selected_makes:
        filtered_df = filtered_df[filtered_df['make'].isin(selected_makes)]
    
    filtered_df = filtered_df[
        (filtered_df['sellingprice'] >= price_range[0]) & 
        (filtered_df['sellingprice'] <= price_range[1]) &
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1])
    ]
    

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Cars", f"{len(filtered_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_price = filtered_df['sellingprice'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Price", f"${avg_price:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_mileage = filtered_df['odometer'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Mileage", f"{avg_mileage:,.0f} mi")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_age = filtered_df['age'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Age", f"{avg_age:.1f} years")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main visualizations based on analysis type
    if analysis_type == "Price Analysis":
        st.subheader("💰 Price Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_price_mileage_scatter(df, selected_makes, price_range, year_range)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_condition_analysis(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.write("**💡 Key Insights:**")
        correlation = filtered_df['odometer'].corr(filtered_df['sellingprice'])
        st.write(f"• Price-Mileage Correlation: {correlation:.3f} (negative correlation indicates higher mileage = lower price)")
        st.write(f"• Price Range: ${filtered_df['sellingprice'].min():,.0f} - ${filtered_df['sellingprice'].max():,.0f}")
        st.write(f"• Most common price range: ${filtered_df['sellingprice'].quantile(0.25):,.0f} - ${filtered_df['sellingprice'].quantile(0.75):,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif analysis_type == "Market Trends":
        st.subheader("📈 Market Trends Analysis")
        
        fig3 = create_time_series(df, selected_makes)
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = create_body_type_analysis(filtered_df)
        st.plotly_chart(fig4, use_container_width=True)
    
    elif analysis_type == "Brand Comparison":
        st.subheader("🏭 Brand Comparison Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig5 = create_brand_comparison(filtered_df, 'sellingprice')
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            fig6 = create_brand_comparison(filtered_df, 'volume')
            st.plotly_chart(fig6, use_container_width=True)
    
    else:  # Geographic Distribution
        st.subheader("🗺️ Geographic Distribution Analysis")
        
        fig7 = create_geographic_distribution(filtered_df)
        if fig7 is not None:
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.error("Unable to create geographic visualization. Please check if state data is available.")
        
        # State-wise summary
        state_summary = filtered_df.groupby('state').agg({
            'sellingprice': ['mean', 'count'],
            'odometer': 'mean'
        }).round(2)
        state_summary.columns = ['Avg Price ($)', 'Sales Count', 'Avg Mileage']
        state_summary = state_summary.sort_values('Sales Count', ascending=False).head(10)
        
        st.subheader("🏆 Top 10 States by Sales Volume")
        st.dataframe(state_summary, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚗 Interactive Car Sales Analytics Dashboard | Built with Streamlit & Plotly</p>
        <p>Features: Interactive filters, dynamic visualizations, real-time updates, and comprehensive market insights</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 