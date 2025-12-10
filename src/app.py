# Stock Dashboard Flask Application
# Demonstrates yfinance integration with caching, error handling, and monitoring

from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime
import traceback
import os
import sys
import logging
from flask_cors import CORS

from .yfinance_wrapper import (
    get_stock_history, get_stock_info, get_multiple_stocks,
    get_latest_candle, get_current_price,
    InvalidTickerError, RateLimitError, NetworkError, resolve_ticker,
    convert_temporal_context_to_dates
)
from .cache_manager import cache_manager
from .logging_config import log_performance_metric
from .query_parser import parse_query
from .pattern_detector import PatternDetector


def create_detailed_multi_stock_summary(executed_actions, response_data, tickers):
    """
    Create detailed, structured summary for multi-stock, multi-intent queries.
    Returns a formatted summary with sections for each stock and action type.
    """
    summary_lines = []

    for ticker in tickers:
        stock_summary = [f"\n📊 {ticker}:"]

        # Add OHLCV information if executed
        if 'ohlcv' in executed_actions and 'ohlcv_summary' in response_data:
            ohlcv_data = response_data['ohlcv_summary'].get(ticker, {})
            if ohlcv_data:
                close_price = ohlcv_data.get('close', 'N/A')
                change_pct = ohlcv_data.get('change_percent')
                if change_pct is not None:
                    change_indicator = "+" if change_pct >= 0 else ""
                    price_info = f"  💰 Price: ₹{close_price:.2f} ({change_indicator}{change_pct:.2f}%)"
                else:
                    price_info = f"  💰 Price: ₹{close_price:.2f}"
                stock_summary.append(price_info)

        # Add pattern information if executed
        if 'pattern_detection' in executed_actions and 'pattern_data' in response_data:
            patterns = response_data['pattern_data'].get('pattern_analysis', {}).get(ticker, [])
            if patterns:
                pattern_names = [p.get('pattern', 'Unknown') for p in patterns]
                stock_summary.append(f"  📈 Patterns Found: {', '.join(pattern_names)}")

                # Add details for each pattern
                for i, pattern in enumerate(patterns, 1):
                    pattern_name = pattern.get('pattern', 'Unknown')
                    confidence = pattern.get('confidence', 0)
                    signal = pattern.get('signal', '')
                    stock_summary.append(f"    {i}. {pattern_name} ({confidence:.1%} confidence) - {signal}")
            else:
                stock_summary.append("  📈 Patterns Found: None")

        # Add latest candle info for context (if available)
        if 'pattern_detection' in executed_actions and 'pattern_data' in response_data:
            latest_candle = response_data['pattern_data'].get('latest_candles', {}).get(ticker, {})
            if latest_candle:
                stock_summary.append(f"  📊 Latest Candle: O:{latest_candle.get('open', 'N/A'):.2f} "
                                   f"H:{latest_candle.get('high', 'N/A'):.2f} "
                                   f"L:{latest_candle.get('low', 'N/A'):.2f} "
                                   f"C:{latest_candle.get('close', 'N/A'):.2f}")

        summary_lines.extend(stock_summary)

    # Add overall statistics
    total_patterns = 0
    if 'pattern_detection' in executed_actions and 'pattern_data' in response_data:
        pattern_analysis = response_data['pattern_data'].get('pattern_analysis', {})
        total_patterns = sum(len(patterns) for patterns in pattern_analysis.values())

    summary_lines.append("\n📋 Summary:")
    summary_lines.append(f"  • Total stocks analyzed: {len(tickers)}")
    summary_lines.append(f"  • Pattern occurrences found: {total_patterns}")
    summary_lines.append(f"  • Actions performed: {', '.join(executed_actions)}")

    return '\n'.join(summary_lines)

# Configure Flask app
app = Flask(__name__,
           template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))

# Enable CORS for all routes
CORS(app)

@app.route('/')
def index():
    """Main dashboard page"""
    logging.debug("DEBUG: index() - Function called to render main dashboard page")
    logging.debug("DEBUG: index() - Rendering template: index.html")
    result = render_template('index.html')
    logging.debug("DEBUG: index() - Template rendered successfully")
    return result

@app.route('/api/stock/<ticker>')
def get_stock(ticker):
    """API endpoint for single stock data"""
    logging.debug(f"DEBUG: get_stock() - Function called with ticker: {ticker}")
    start_time = datetime.utcnow()
    logging.debug(f"DEBUG: get_stock() - Start time recorded: {start_time}")

    # Get exchange parameter from query string
    exchange = request.args.get('exchange', 'nse')  # default to nse
    logging.debug(f"DEBUG: get_stock() - Exchange parameter: {exchange}")

    try:
        logging.debug(f"DEBUG: get_stock() - Attempting to get cached data for ticker: {ticker}, period: 1y")
        # Try cache first
        cached_data = cache_manager.get(ticker, period="1y")
        logging.debug(f"DEBUG: get_stock() - Cache lookup result: {'found' if cached_data is not None else 'not found'}")

        if cached_data is not None:
            logging.debug("DEBUG: get_stock() - Using cached data")
            # Return cached data
            data = {str(k): v for k, v in cached_data.to_dict('index').items()}
            logging.debug(f"DEBUG: get_stock() - Converted cached data to dict format, keys count: {len(data)}")
            result = jsonify({
                'success': True,
                'data': data,
                'source': 'cache',
                'timestamp': datetime.utcnow().isoformat()
            })
            logging.debug("DEBUG: get_stock() - Returning cached data response")
            return result

        logging.debug(f"DEBUG: get_stock() - Cache miss, fetching from API for ticker: {ticker}")
        # Fetch from API with exchange context
        df = get_stock_history(ticker, period="1y", exchange=exchange)
        logging.debug(f"DEBUG: get_stock() - API fetch successful, dataframe shape: {df.shape}")

        # Cache the result
        logging.debug(f"DEBUG: get_stock() - Caching fetched data for ticker: {ticker}")
        cache_manager.put(ticker, df, period="1y")
        logging.debug("DEBUG: get_stock() - Data cached successfully")

        # Convert to JSON-serializable format
        data = {str(k): v for k, v in df.to_dict('index').items()}
        logging.debug(f"DEBUG: get_stock() - Converted API data to dict format, keys count: {len(data)}")

        # Log performance
        duration = (datetime.utcnow() - start_time).total_seconds()
        log_performance_metric("single_stock_fetch", duration)
        logging.debug(f"DEBUG: get_stock() - Performance logged: {duration} seconds")

        result = jsonify({
            'success': True,
            'data': data,
            'source': 'api',
            'timestamp': datetime.utcnow().isoformat()
        })
        logging.debug("DEBUG: get_stock() - Returning API data response")
        return result

    except InvalidTickerError as e:
        logging.debug(f"DEBUG: get_stock() - InvalidTickerError caught for ticker: {ticker}, error: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Invalid ticker symbol',
            'message': str(e)
        }), 400
        logging.debug("DEBUG: get_stock() - Returning InvalidTickerError response")
        return result

    except RateLimitError as e:
        logging.debug(f"DEBUG: get_stock() - RateLimitError caught for ticker: {ticker}, error: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Rate limit exceeded',
            'message': 'Please try again later'
        }), 429
        logging.debug("DEBUG: get_stock() - Returning RateLimitError response")
        return result

    except NetworkError as e:
        logging.debug(f"DEBUG: get_stock() - NetworkError caught for ticker: {ticker}, error: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Network error',
            'message': 'Unable to fetch data. Please try again.'
        }), 503
        logging.debug("DEBUG: get_stock() - Returning NetworkError response")
        return result

    except Exception as e:
        logging.error(f"Unexpected error for ticker {ticker}: {traceback.format_exc()}")
        logging.debug(f"DEBUG: get_stock() - Unexpected Exception caught for ticker: {ticker}, type: {type(e).__name__}")
        result = jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
        logging.debug("DEBUG: get_stock() - Returning unexpected error response")
        return result

@app.route('/api/stocks/batch', methods=['POST'])
def get_stocks_batch():
    """API endpoint for multiple stocks"""
    logging.debug("DEBUG: get_stocks_batch() - Function called for batch stock request")
    start_time = datetime.utcnow()
    logging.debug(f"DEBUG: get_stocks_batch() - Start time recorded: {start_time}")

    try:
        logging.debug("DEBUG: get_stocks_batch() - Parsing JSON request data")
        data = request.get_json()
        logging.debug(f"DEBUG: get_stocks_batch() - Request data parsed: {data is not None}")

        if not data or 'tickers' not in data:
            logging.debug("DEBUG: get_stocks_batch() - Missing tickers parameter in request")
            result = jsonify({
                'success': False,
                'error': 'Missing tickers parameter'
            }), 400
            logging.debug("DEBUG: get_stocks_batch() - Returning missing tickers error response")
            return result

        tickers = data['tickers']
        exchange = data.get('exchange', 'nse')  # default to nse
        logging.debug(f"DEBUG: get_stocks_batch() - Tickers extracted: {tickers}, exchange: {exchange}")

        if not isinstance(tickers, list) or len(tickers) > 20:
            logging.debug(f"DEBUG: get_stocks_batch() - Invalid tickers format or count. Type: {type(tickers)}, Length: {len(tickers) if isinstance(tickers, list) else 'N/A'}")
            result = jsonify({
                'success': False,
                'error': 'Tickers must be a list with max 20 items'
            }), 400
            logging.debug("DEBUG: get_stocks_batch() - Returning invalid tickers error response")
            return result

        logging.debug(f"DEBUG: get_stocks_batch() - Fetching batch data for {len(tickers)} tickers")
        # Get batch data
        results = get_multiple_stocks(tickers, period="1y", exchange=exchange)
        logging.debug(f"DEBUG: get_stocks_batch() - Batch fetch completed, results count: {len(results)}")

        # Convert to JSON-serializable format
        logging.debug("DEBUG: get_stocks_batch() - Converting results to JSON-serializable format")
        response_data = {}
        for ticker, df in results.items():
            response_data[ticker] = {str(k): v for k, v in df.to_dict('index').items()}
            logging.debug(f"DEBUG: get_stocks_batch() - Converted data for ticker: {ticker}, shape: {df.shape}")

        # Log performance
        duration = (datetime.utcnow() - start_time).total_seconds()
        log_performance_metric("batch_stocks_fetch", duration, f"seconds_for_{len(tickers)}_stocks")
        logging.debug(f"DEBUG: get_stocks_batch() - Performance logged: {duration} seconds for {len(tickers)} stocks")

        result = jsonify({
            'success': True,
            'data': response_data,
            'count': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
        logging.debug("DEBUG: get_stocks_batch() - Returning successful batch response")
        return result

    except Exception as e:
        app.logger.error(f"Batch request error: {traceback.format_exc()}")
        logging.debug(f"DEBUG: get_stocks_batch() - Exception caught: {type(e).__name__}: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
        logging.debug("DEBUG: get_stocks_batch() - Returning batch error response")
        return result

@app.route('/api/stock/<ticker>/info')
def get_stock_info_endpoint(ticker):
    """API endpoint for stock company information"""
    logging.debug(f"DEBUG: get_stock_info_endpoint() - Function called with ticker: {ticker}")

    try:
        logging.debug(f"DEBUG: get_stock_info_endpoint() - Fetching stock info for ticker: {ticker}")
        info = get_stock_info(ticker)
        logging.debug(f"DEBUG: get_stock_info_endpoint() - Stock info fetched successfully, keys: {list(info.keys()) if info else 'None'}")

        result = jsonify({
            'success': True,
            'data': info,
            'timestamp': datetime.utcnow().isoformat()
        })
        logging.debug("DEBUG: get_stock_info_endpoint() - Returning successful stock info response")
        return result

    except InvalidTickerError as e:
        logging.debug(f"DEBUG: get_stock_info_endpoint() - InvalidTickerError caught for ticker: {ticker}, error: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Invalid ticker symbol',
            'message': str(e)
        }), 400
        logging.debug("DEBUG: get_stock_info_endpoint() - Returning InvalidTickerError response")
        return result

    except Exception as e:
        app.logger.error(f"Info request error for {ticker}: {traceback.format_exc()}")
        logging.debug(f"DEBUG: get_stock_info_endpoint() - Exception caught: {type(e).__name__}: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
        logging.debug("DEBUG: get_stock_info_endpoint() - Returning error response")
        return result

@app.route('/api/stock/<ticker>/latest')
def get_latest_candle_endpoint(ticker):
    """API endpoint for latest candle/price data"""
    logging.debug(f"DEBUG: get_latest_candle_endpoint() - Function called with ticker: {ticker}")

    # Get exchange parameter from query string
    exchange = request.args.get('exchange', 'nse')  # default to nse
    logging.debug(f"DEBUG: get_latest_candle_endpoint() - Exchange parameter: {exchange}")

    try:
        logging.debug(f"DEBUG: get_latest_candle_endpoint() - Fetching latest candle for ticker: {ticker}")
        candle = get_latest_candle(ticker, exchange=exchange, data_source="smart_fallback", context="latest")
        logging.debug(f"DEBUG: get_latest_candle_endpoint() - Latest candle fetched successfully: {candle}")

        result = jsonify({
            'success': True,
            'data': candle,
            'timestamp': datetime.utcnow().isoformat()
        })
        logging.debug("DEBUG: get_latest_candle_endpoint() - Returning successful latest candle response")
        return result

    except InvalidTickerError as e:
        logging.debug(f"DEBUG: get_latest_candle_endpoint() - InvalidTickerError caught for ticker: {ticker}, error: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Invalid ticker symbol',
            'message': str(e)
        }), 400
        logging.debug("DEBUG: get_latest_candle_endpoint() - Returning InvalidTickerError response")
        return result

    except Exception as e:
        app.logger.error(f"Latest candle request error for {ticker}: {traceback.format_exc()}")
        logging.debug(f"DEBUG: get_latest_candle_endpoint() - Exception caught: {type(e).__name__}: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
        logging.debug("DEBUG: get_latest_candle_endpoint() - Returning error response")
        return result

@app.route('/api/stock/<ticker>/price')
def get_current_price_endpoint(ticker):
    """API endpoint for current price (lightweight)"""
    logging.debug(f"DEBUG: get_current_price_endpoint() - Function called with ticker: {ticker}")

    # Get exchange parameter from query string
    exchange = request.args.get('exchange', 'nse')  # default to nse
    logging.debug(f"DEBUG: get_current_price_endpoint() - Exchange parameter: {exchange}")

    try:
        logging.debug(f"DEBUG: get_current_price_endpoint() - Fetching current price for ticker: {ticker}")
        price = get_current_price(ticker, exchange=exchange)
        logging.debug(f"DEBUG: get_current_price_endpoint() - Current price fetched successfully: {price}")

        result = jsonify({
            'success': True,
            'data': price,
            'timestamp': datetime.utcnow().isoformat()
        })
        logging.debug("DEBUG: get_current_price_endpoint() - Returning successful current price response")
        return result

    except InvalidTickerError as e:
        logging.debug(f"DEBUG: get_current_price_endpoint() - InvalidTickerError caught for ticker: {ticker}, error: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Invalid ticker symbol',
            'message': str(e)
        }), 400
        logging.debug("DEBUG: get_current_price_endpoint() - Returning InvalidTickerError response")
        return result

    except Exception as e:
        app.logger.error(f"Current price request error for {ticker}: {traceback.format_exc()}")
        logging.debug(f"DEBUG: get_current_price_endpoint() - Exception caught: {type(e).__name__}: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
        logging.debug("DEBUG: get_current_price_endpoint() - Returning error response")
        return result

@app.route('/api/cache/stats')
def get_cache_stats():
    """API endpoint for cache statistics"""
    logging.debug("DEBUG: get_cache_stats() - Function called to retrieve cache statistics")

    try:
        logging.debug("DEBUG: get_cache_stats() - Fetching cache statistics")
        stats = cache_manager.get_stats()
        logging.debug(f"DEBUG: get_cache_stats() - Cache stats retrieved: {stats}")

        result = jsonify({
            'success': True,
            'data': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        logging.debug("DEBUG: get_cache_stats() - Returning successful cache stats response")
        return result

    except Exception as e:
        app.logger.error(f"Cache stats error: {traceback.format_exc()}")
        logging.debug(f"DEBUG: get_cache_stats() - Exception caught: {type(e).__name__}: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
        logging.debug("DEBUG: get_cache_stats() - Returning error response")
        return result

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """API endpoint to clear cache"""
    logging.debug("DEBUG: clear_cache() - Function called to clear cache")

    try:
        logging.debug("DEBUG: clear_cache() - Parsing request JSON data")
        data = request.get_json() or {}
        ticker = data.get('ticker')
        logging.debug(f"DEBUG: clear_cache() - Ticker to clear: {ticker if ticker else 'all tickers'}")

        logging.debug(f"DEBUG: clear_cache() - Clearing cache for {'ticker: ' + ticker if ticker else 'all tickers'}")
        cache_manager.clear(ticker)
        logging.debug("DEBUG: clear_cache() - Cache cleared successfully")

        result = jsonify({
            'success': True,
            'message': f'Cache cleared for {ticker if ticker else "all tickers"}',
            'timestamp': datetime.utcnow().isoformat()
        })
        logging.debug("DEBUG: clear_cache() - Returning successful cache clear response")
        return result

    except Exception as e:
        app.logger.error(f"Cache clear error: {traceback.format_exc()}")
        logging.debug(f"DEBUG: clear_cache() - Exception caught: {type(e).__name__}: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
        logging.debug("DEBUG: clear_cache() - Returning error response")
        return result

@app.route('/api/query', methods=['POST'])
def natural_language_query():
    """API endpoint for natural language queries"""
    logging.debug("DEBUG: natural_language_query() - Function called for natural language query processing")
    start_time = datetime.utcnow()
    logging.debug(f"DEBUG: natural_language_query() - Start time recorded: {start_time}")

    try:
        logging.debug("DEBUG: natural_language_query() - Parsing JSON request data")
        data = request.get_json()
        logging.debug(f"DEBUG: natural_language_query() - Request data parsed: {data is not None}")

        if not data or 'query' not in data:
            logging.debug("DEBUG: natural_language_query() - Missing query parameter in request")
            result = jsonify({
                'success': False,
                'error': 'Missing query parameter',
                'message': 'Please provide a natural language query'
            }), 400
            logging.debug("DEBUG: natural_language_query() - Returning missing query error response")
            return result

        query = data['query']
        exchange = data.get('exchange', 'nse')  # default to nse
        logging.debug(f"DEBUG: natural_language_query() - Query extracted: '{query}', exchange: {exchange}")

        if not query or not isinstance(query, str):
            logging.debug(f"DEBUG: natural_language_query() - Invalid query format. Query: {query}, Type: {type(query)}")
            result = jsonify({
                'success': False,
                'error': 'Invalid query',
                'message': 'Query must be a non-empty string'
            }), 400
            logging.debug("DEBUG: natural_language_query() - Returning invalid query error response")
            return result

        # Parse the natural language query
        logging.debug(f"DEBUG: natural_language_query() - Parsing query with spacy=False, validate_ticker=True, exchange={exchange}")
        parsed_result = parse_query(query, use_spacy=False, validate_ticker=True, exchange=exchange)
        logging.debug(f"DEBUG: natural_language_query() - Query parsing completed. Result keys: {list(parsed_result.keys())}")

        # Check if parsing was successful
        if parsed_result.get('error'):
            logging.debug(f"DEBUG: natural_language_query() - Query parsing failed: {parsed_result.get('message')}")
            result = jsonify({
                'success': False,
                'error': 'Query parsing failed',
                'message': parsed_result.get('message'),
                'suggestions': parsed_result.get('suggestions', []),
                'parsed_query': parsed_result
            }), 400
            logging.debug("DEBUG: natural_language_query() - Returning query parsing error response")
            return result

        tickers = parsed_result.get('tickers', [])
        ticker = parsed_result['ticker']  # Backward compatibility
        timeframes = parsed_result.get('timeframes', [])
        timeframe = parsed_result['timeframe']  # Backward compatibility
        query_types = parsed_result.get('query_types', [])
        query_type = parsed_result['query_type']  # Primary intent for backward compatibility
        pattern = parsed_result['pattern']
        logging.debug(f"DEBUG: natural_language_query() - Parsed components - tickers: {tickers}, ticker: {ticker}, timeframes: {timeframes}, timeframe: {timeframe}, query_types: {query_types}, query_type: {query_type}, pattern: {pattern}")

        # Resolve tickers based on exchange selection for consistent display
        resolved_tickers = [resolve_ticker(t, exchange) for t in tickers]
        resolved_ticker = resolved_tickers[0] if resolved_tickers else None  # Backward compatibility

        # Update parsed_result with resolved tickers for UI display
        parsed_result['tickers'] = resolved_tickers
        parsed_result['ticker'] = resolved_ticker

        logging.debug("DEBUG: natural_language_query() - Preparing response data structure")
        response_data = {
            'success': True,
            'query': query,
            'parsed': {
                'tickers': resolved_tickers,
                'ticker': resolved_ticker,  # Backward compatibility
                'timeframes': timeframes,
                'timeframe': timeframe,  # Backward compatibility
                'query_types': query_types,
                'query_type': query_type,  # Backward compatibility
                'pattern': pattern,
                'confidence': parsed_result['confidence']
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        logging.debug(f"DEBUG: natural_language_query() - Response data initialized with query_types: {query_types}, tickers: {resolved_tickers}, timeframes: {timeframes}")

        # Execute the appropriate action based on query types (can be multiple)
        executed_actions = []

        if "current_price" in query_types:
            logging.debug("DEBUG: natural_language_query() - Processing current_price query type for multiple tickers")
            # For current price queries, use the lightweight current price endpoint
            try:
                price_results = {}
                failed_tickers = []

                for ticker_symbol in resolved_tickers:
                    try:
                        logging.debug(f"DEBUG: natural_language_query() - Fetching current price for ticker: {ticker_symbol}, exchange: {exchange}")
                        price_data = get_current_price(ticker_symbol, exchange=exchange)
                        logging.debug(f"DEBUG: natural_language_query() - Current price fetched for {ticker_symbol}: {price_data}")

                        price_results[ticker_symbol] = price_data

                    except Exception as e:
                        logging.debug(f"DEBUG: natural_language_query() - Failed to fetch current price for {ticker_symbol}: {str(e)}")
                        failed_tickers.append(ticker_symbol)
                        continue

                if not price_results:
                    logging.debug("DEBUG: natural_language_query() - No current price data available for any ticker")
                    result = jsonify({
                        'success': False,
                        'error': 'No data available',
                        'message': f'Could not fetch current price data for any of the requested tickers: {resolved_tickers}',
                        'parsed_query': parsed_result
                    }), 404
                    logging.debug("DEBUG: natural_language_query() - Returning no current price data available response")
                    return result

                executed_actions.append('current_price')
                response_data['current_price_data'] = price_results

                successful_tickers = list(price_results.keys())

                # Create summary message for current prices
                price_summaries = []
                for ticker_symbol in successful_tickers:
                    price_info = price_results[ticker_symbol]
                    price = price_info.get('price', 'N/A')
                    currency = price_info.get('currency', '₹')
                    if isinstance(price, (int, float)):
                        price_summaries.append(f"{ticker_symbol}: {currency}{price:.2f}")
                    else:
                        price_summaries.append(f"{ticker_symbol}: {price}")

                response_data['current_price_summary'] = price_summaries

                if len(successful_tickers) == 1:
                    # Single stock - simple format
                    response_data['message'] = f"Current price for {successful_tickers[0]}: {price_summaries[0]}"
                else:
                    # Multi-stock - detailed format
                    response_data['message'] = f"Current prices retrieved for {len(successful_tickers)} stocks:\n"
                    response_data['message'] += '\n'.join(f"  • {summary}" for summary in price_summaries)

                if failed_tickers:
                    response_data['message'] += f". Failed to fetch data for: {', '.join(failed_tickers)}"

                logging.debug(f"DEBUG: natural_language_query() - Current price response data prepared for {len(successful_tickers)} tickers")

            except Exception as e:
                app.logger.error(f"Current price fetch error: {traceback.format_exc()}")
                logging.debug(f"DEBUG: natural_language_query() - Exception in current price fetch: {type(e).__name__}: {str(e)}")
                result = jsonify({
                    'success': False,
                    'error': 'Data fetch failed',
                    'message': f'Could not fetch current price data: {str(e)}',
                    'parsed_query': parsed_result
                }), 500
                logging.debug("DEBUG: natural_language_query() - Returning current price fetch error response")
                return result

        if "ohlcv" in query_types:
            logging.debug("DEBUG: natural_language_query() - Processing OHLCV query type for multiple tickers and timeframes")
            temporal_context = parsed_result.get('temporal_context')
            specific_date = parsed_result.get('date')
            logging.debug(f"DEBUG: natural_language_query() - Temporal context: {temporal_context}, specific date: {specific_date}")

            # For multi-timeframe queries, use the first timeframe as primary for now
            # TODO: Implement proper multi-timeframe support
            primary_timeframe = timeframes[0] if timeframes else timeframe
            logging.debug(f"DEBUG: natural_language_query() - Initial primary timeframe: {primary_timeframe}")

            # Handle temporal contexts that modify timeframe/period
            is_intraday_today = False
            ohlcv_period = None
            ohlcv_start_date = None
            ohlcv_end_date = None

            # Check for specific date first (takes precedence over temporal context)
            if specific_date:
                # For specific date queries, fetch data around that date
                from datetime import timedelta
                date_obj = datetime.strptime(specific_date, '%Y-%m-%d')
                ohlcv_start_date = (date_obj - timedelta(days=5)).strftime('%Y-%m-%d')
                ohlcv_end_date = (date_obj + timedelta(days=5)).strftime('%Y-%m-%d')
                logging.debug(f"DEBUG: natural_language_query() - Specific date '{specific_date}' converted to range: {ohlcv_start_date} to {ohlcv_end_date}")

            elif temporal_context:
                # Convert temporal context to actual date ranges
                ohlcv_start_date, ohlcv_end_date = convert_temporal_context_to_dates(temporal_context)
                logging.debug(f"DEBUG: natural_language_query() - Temporal context '{temporal_context}' converted to date range: {ohlcv_start_date} to {ohlcv_end_date}")

                # Special handling for today
                if temporal_context == "today" and not primary_timeframe:
                    # "today's data" with no timeframe → fetch intraday data for current day
                    primary_timeframe = "5m"  # Default to 5-minute intervals for today
                    is_intraday_today = True
                    logging.debug(f"DEBUG: natural_language_query() - 'today' context detected, defaulting to intraday timeframe: {primary_timeframe}")

                # For temporal contexts with specific date ranges, use a period that covers the range plus some buffer
                if ohlcv_start_date and ohlcv_end_date:
                    # Calculate period needed to cover the date range plus buffer
                    start_dt = datetime.strptime(ohlcv_start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(ohlcv_end_date, '%Y-%m-%d')
                    days_needed = (end_dt - start_dt).days + 7  # Add 1 week buffer
                    if days_needed <= 7:
                        ohlcv_period = "1wk"
                    elif days_needed <= 31:
                        ohlcv_period = "1mo"
                    elif days_needed <= 90:
                        ohlcv_period = "3mo"
                    else:
                        ohlcv_period = "1y"
                    logging.debug(f"DEBUG: natural_language_query() - Date range requires {days_needed} days, using period: {ohlcv_period}")
            else:
                # No temporal context - use default period logic
                if len(resolved_tickers) > 1:
                    # Multi-ticker comparison - use 3 months instead of default 1 year
                    ohlcv_period = "3mo"
                    logging.debug(f"DEBUG: natural_language_query() - Multi-ticker query detected ({len(resolved_tickers)} tickers), using shorter period: {ohlcv_period}")
                elif primary_timeframe == "1d":
                    # Daily chart requests - use 6 months instead of 1 year for better performance
                    ohlcv_period = "6mo"
                    logging.debug(f"DEBUG: natural_language_query() - Daily timeframe detected, using medium period: {ohlcv_period}")

            logging.debug(f"DEBUG: natural_language_query() - Final timeframe: {primary_timeframe}, period: {ohlcv_period}, start_date: {ohlcv_start_date}, end_date: {ohlcv_end_date}")

            # Fetch OHLCV data for each ticker
            try:
                ohlcv_results = {}
                summary_results = {}
                failed_tickers = []

                for ticker_symbol in resolved_tickers:
                    try:
                        logging.debug(f"DEBUG: natural_language_query() - Fetching OHLCV data for ticker: {ticker_symbol}, timeframe: {primary_timeframe}, period: {ohlcv_period}, start_date: {ohlcv_start_date}, end_date: {ohlcv_end_date}, exchange: {exchange}")
                        ohlcv_df = get_stock_history(ticker_symbol, period=ohlcv_period, start_date=ohlcv_start_date, end_date=ohlcv_end_date,
                                                   timeframe=primary_timeframe, exchange=exchange, data_source="smart_fallback", context="historical")
                        logging.debug(f"DEBUG: natural_language_query() - OHLCV data fetched for {ticker_symbol}, dataframe shape: {ohlcv_df.shape}")

                        if ohlcv_df.empty:
                            logging.debug(f"DEBUG: natural_language_query() - No OHLCV data available for {ticker_symbol}")
                            failed_tickers.append(ticker_symbol)
                            continue

                        # Handle different data formats based on query type
                        if is_intraday_today and len(resolved_tickers) == 1:
                            # For "today's data" with intraday timeframe, return full data for charting
                            logging.debug(f"DEBUG: natural_language_query() - Returning full intraday data for today's {ticker_symbol} data")
                            ohlcv_data = {str(k): v for k, v in ohlcv_df.to_dict('index').items()}
                            ohlcv_results[ticker_symbol] = ohlcv_data

                            # For intraday today data, don't create summary - return full data
                            summary_results[ticker_symbol] = {
                                "data_points": len(ohlcv_df),
                                "timeframe": primary_timeframe,
                                "period": ohlcv_period,
                                "temporal_context": temporal_context,
                                "full_intraday_data": True
                            }
                        else:
                            # Standard OHLCV processing - convert to JSON and create summary
                            ohlcv_data = {str(k): v for k, v in ohlcv_df.to_dict('index').items()}
                            ohlcv_results[ticker_symbol] = ohlcv_data

                            # Build latest candle summary including change vs previous close
                            # For specific date queries, use the data for that date, otherwise use latest
                            if specific_date:
                                # Find the specific date's data
                                specific_date_data = ohlcv_df[ohlcv_df.index.strftime('%Y-%m-%d') == specific_date]
                                if not specific_date_data.empty:
                                    target_row = specific_date_data.iloc[0]
                                    target_index = target_row.name
                                    latest_close = float(target_row["Close"])
                                    summary = {
                                        "timestamp": target_index.strftime('%Y-%m-%d'),
                                        "open": float(target_row["Open"]),
                                        "high": float(target_row["High"]),
                                        "low": float(target_row["Low"]),
                                        "close": latest_close,
                                        "volume": int(target_row["Volume"]),
                                    }
                                    logging.debug(f"DEBUG: natural_language_query() - Using specific date data for {ticker_symbol}: {specific_date}")
                                else:
                                    # Fallback to latest if specific date not found
                                    latest_index = list(ohlcv_df.index)[-1]
                                    latest_row = ohlcv_df.iloc[-1]
                                    latest_close = float(latest_row["Close"])
                                    summary = {
                                        "timestamp": latest_index.strftime('%Y-%m-%d'),
                                        "open": float(latest_row["Open"]),
                                        "high": float(latest_row["High"]),
                                        "low": float(latest_row["Low"]),
                                        "close": latest_close,
                                        "volume": int(latest_row["Volume"]),
                                    }
                                    logging.debug(f"DEBUG: natural_language_query() - Specific date {specific_date} not found for {ticker_symbol}, using latest")
                            else:
                                latest_index = list(ohlcv_df.index)[-1]
                                latest_row = ohlcv_df.iloc[-1]
                                latest_close = float(latest_row["Close"])
                                summary = {
                                    "timestamp": latest_index.strftime('%Y-%m-%d'),
                                    "open": float(latest_row["Open"]),
                                    "high": float(latest_row["High"]),
                                    "low": float(latest_row["Low"]),
                                    "close": latest_close,
                                    "volume": int(latest_row["Volume"]),
                                }

                        # Calculate change vs previous close
                        change = None
                        change_pct = None
                        prev_close = None

                        if specific_date and 'specific_date_data' in locals() and not specific_date_data.empty:
                            # For specific date, find the previous trading day in the dataset
                            target_idx = ohlcv_df.index.get_loc(target_index)
                            if target_idx > 0:
                                prev_row = ohlcv_df.iloc[target_idx - 1]
                                prev_close = float(prev_row["Close"])
                                change = latest_close - prev_close
                                change_pct = (change / prev_close * 100.0) if prev_close != 0 else 0.0
                        elif len(ohlcv_df) >= 2:
                            # For non-specific date queries, use the standard logic
                            prev_close = float(ohlcv_df.iloc[-2]["Close"])
                            change = latest_close - prev_close
                            change_pct = (change / prev_close * 100.0) if prev_close != 0 else 0.0

                        if change is not None and change_pct is not None:
                            summary.update({
                                "previous_close": round(prev_close, 4),
                                "change": round(change, 4),
                                "change_percent": round(change_pct, 4),
                            })

                        summary_results[ticker_symbol] = summary
                        logging.debug(f"DEBUG: natural_language_query() - Processed OHLCV data for {ticker_symbol}")

                    except Exception as e:
                        logging.debug(f"DEBUG: natural_language_query() - Failed to fetch OHLCV data for {ticker_symbol}: {str(e)}")
                        failed_tickers.append(ticker_symbol)
                        continue

                if not ohlcv_results:
                    logging.debug("DEBUG: natural_language_query() - No OHLCV data available for any ticker")
                    result = jsonify({
                        'success': False,
                        'error': 'No data available',
                        'message': f'Could not fetch OHLCV data for any of the requested tickers: {resolved_tickers}',
                        'parsed_query': parsed_result
                    }), 404
                    logging.debug("DEBUG: natural_language_query() - Returning no data available response")
                    return result

                executed_actions.append('ohlcv')
                response_data['ohlcv_data'] = ohlcv_results
                response_data['ohlcv_summary'] = summary_results

                successful_tickers = list(ohlcv_results.keys())

                # Create detailed per-stock summary for OHLCV
                ohlcv_stock_summaries = []
                message_parts = []
                for ticker_symbol in successful_tickers:
                    summary = summary_results[ticker_symbol]
                    latest_close = summary["close"]
                    change_pct = summary.get("change_percent", 0)
                    change_indicator = "+" if change_pct >= 0 else ""

                    ohlcv_stock_summaries.append(
                        f"{ticker_symbol}: ₹{latest_close:.2f} ({change_indicator}{change_pct:.2f}%)"
                    )
                    message_parts.append(f"{ticker_symbol}: ₹{latest_close:.2f}")

                response_data['ohlcv_summary']['stock_summaries'] = ohlcv_stock_summaries

                if len(successful_tickers) == 1 and is_intraday_today:
                    # Today's intraday data - special message
                    summary = summary_results[successful_tickers[0]]
                    data_points = summary.get("data_points", 0)
                    response_data['message'] = (
                        f"Retrieved today's intraday data for {successful_tickers[0]} "
                        f"({data_points} {primary_timeframe} candles available for charting)"
                    )
                elif len(successful_tickers) == 1:
                    # Single stock - include temporal context in message if available
                    if temporal_context and ohlcv_start_date and ohlcv_end_date:
                        date_range_text = f"from {ohlcv_start_date} to {ohlcv_end_date}"
                        response_data['message'] = (
                            f"Retrieved OHLCV data for {successful_tickers[0]} on {primary_timeframe} timeframe "
                            f"for {temporal_context.replace('_', ' ')} ({date_range_text}) - {message_parts[0]}"
                        )
                    else:
                        response_data['message'] = (
                            f"Retrieved OHLCV data for {successful_tickers[0]} on {primary_timeframe} timeframe "
                            f"({message_parts[0]})"
                        )
                else:
                    # Multi-stock - use detailed breakdown with period indication
                    if temporal_context and ohlcv_start_date and ohlcv_end_date:
                        period_text = f"{temporal_context.replace('_', ' ')} ({ohlcv_start_date} to {ohlcv_end_date})"
                    else:
                        period_text = "recent 3 months" if ohlcv_period == "3mo" else f"{ohlcv_period} period"
                    response_data['message'] = f"Comparing {len(successful_tickers)} stocks on {primary_timeframe} timeframe ({period_text}):\n"
                    response_data['message'] += '\n'.join(f"  • {summary}" for summary in ohlcv_stock_summaries)

                if failed_tickers:
                    response_data['message'] += f". Failed to fetch data for: {', '.join(failed_tickers)}"

                logging.debug(f"DEBUG: natural_language_query() - OHLCV response data prepared for {len(successful_tickers)} tickers")

            except Exception as e:
                app.logger.error(f"OHLCV fetch error: {traceback.format_exc()}")
                logging.debug(f"DEBUG: natural_language_query() - Exception in OHLCV fetch: {type(e).__name__}: {str(e)}")
                result = jsonify({
                    'success': False,
                    'error': 'Data fetch failed',
                    'message': f'Could not fetch OHLCV data: {str(e)}',
                    'parsed_query': parsed_result
                }), 500
                logging.debug("DEBUG: natural_language_query() - Returning OHLCV fetch error response")
                return result

        if "pattern" in query_types:
            logging.debug("DEBUG: natural_language_query() - Processing pattern detection query type for multiple tickers and timeframes")
            # For multi-timeframe queries, use the first timeframe as primary for pattern analysis
            pattern_timeframe = timeframes[0] if timeframes else timeframe
            temporal_context = parsed_result.get('temporal_context')
            specific_date = parsed_result.get('date')
            logging.debug(f"DEBUG: natural_language_query() - Using timeframe: {pattern_timeframe} for pattern detection, temporal_context: {temporal_context}, specific_date: {specific_date}")

            # Detect candlestick patterns for each ticker
            try:
                pattern_results_all = {}
                ohlcv_data_all = {}
                latest_candles_all = {}
                failed_tickers = []

                for ticker_symbol in resolved_tickers:
                    try:
                        # Handle temporal contexts and specific dates for pattern analysis
                        pattern_start_date = None
                        pattern_end_date = None
                        pattern_period = None

                        # Check for specific date first (takes precedence over temporal context)
                        if specific_date:
                            # For specific date queries, fetch data around that date for pattern context
                            from datetime import timedelta
                            date_obj = datetime.strptime(specific_date, '%Y-%m-%d')
                            pattern_start_date = (date_obj - timedelta(days=10)).strftime('%Y-%m-%d')
                            pattern_end_date = (date_obj + timedelta(days=10)).strftime('%Y-%m-%d')
                            logging.debug(f"DEBUG: natural_language_query() - Specific date '{specific_date}' converted to range: {pattern_start_date} to {pattern_end_date}")

                        elif temporal_context:
                            # Convert temporal context to actual date ranges for pattern analysis
                            pattern_start_date, pattern_end_date = convert_temporal_context_to_dates(temporal_context)
                            logging.debug(f"DEBUG: natural_language_query() - Temporal context '{temporal_context}' converted to date range: {pattern_start_date} to {pattern_end_date}")

                            # For temporal contexts, prioritize daily timeframe unless explicit intraday timeframe is specified
                            # If timeframe came from "1 month" parsing and we have temporal context, use daily instead
                            if pattern_timeframe == "1mo" and temporal_context:
                                # Ambiguous case: "1 month" could mean timeframe or temporal context
                                # Prioritize temporal context for clearer intent
                                pattern_timeframe = "1d"  # Default to daily candles for temporal contexts
                                logging.debug(f"DEBUG: natural_language_query() - Ambiguous '1mo' timeframe with temporal context, defaulting to daily: {pattern_timeframe}")
                            elif not pattern_timeframe or pattern_timeframe not in ['1d', '1w', '1h', '5m', '15m', '30m']:
                                # No explicit timeframe or ambiguous timeframe, default to daily for temporal contexts
                                pattern_timeframe = "1d"
                                logging.debug(f"DEBUG: natural_language_query() - Defaulting to daily timeframe for temporal context: {pattern_timeframe}")

                            # Calculate period needed to cover the date range plus some buffer for patterns
                            if pattern_start_date and pattern_end_date:
                                start_dt = datetime.strptime(pattern_start_date, '%Y-%m-%d')
                                end_dt = datetime.strptime(pattern_end_date, '%Y-%m-%d')
                                days_needed = (end_dt - start_dt).days + 14  # Add 2 week buffer for pattern context
                                if days_needed <= 7:
                                    pattern_period = "1wk"
                                elif days_needed <= 31:
                                    pattern_period = "1mo"
                                elif days_needed <= 90:
                                    pattern_period = "3mo"
                                else:
                                    pattern_period = "1y"
                                logging.debug(f"DEBUG: natural_language_query() - Date range requires {days_needed} days, using period: {pattern_period}")

                        # Fetch OHLCV data aligned to timeframe for pattern detection (using history for trend)
                        logging.debug(f"DEBUG: natural_language_query() - Fetching OHLCV data for pattern detection, ticker: {ticker_symbol}, timeframe: {pattern_timeframe}, start_date: {pattern_start_date}, end_date: {pattern_end_date}, period: {pattern_period}")

                        if pattern_start_date and pattern_end_date:
                            # Use date range for temporal context or specific date queries
                            ohlcv_df = get_stock_history(ticker_symbol, start_date=pattern_start_date, end_date=pattern_end_date,
                                                       timeframe=pattern_timeframe, exchange=exchange,
                                                       data_source="smart_fallback", context="pattern")
                        else:
                            # Use period-based fetching (original logic for timeframe-only queries)
                            # Adjust period based on timeframe to get enough candles but not too much
                            if not pattern_period:
                                pattern_period = "1mo" # Default for intraday/daily
                                if pattern_timeframe == "1w":
                                    pattern_period = "1y"
                                elif pattern_timeframe == "1mo":
                                    pattern_period = "2y"

                            ohlcv_df = get_stock_history(ticker_symbol, period=pattern_period, timeframe=pattern_timeframe, exchange=exchange,
                                                       data_source="smart_fallback", context="pattern")
                        logging.debug(f"DEBUG: natural_language_query() - OHLCV data fetched for patterns, ticker: {ticker_symbol}, dataframe shape: {ohlcv_df.shape}")

                        if ohlcv_df.empty:
                            logging.debug(f"DEBUG: natural_language_query() - No OHLCV data available for pattern detection for {ticker_symbol}")
                            failed_tickers.append(ticker_symbol)
                            continue

                        # Initialize pattern detector
                        logging.debug(f"DEBUG: natural_language_query() - Initializing PatternDetector for {ticker_symbol}")
                        detector = PatternDetector()

                        # Analyze patterns using full history for trend context
                        logging.debug(f"DEBUG: natural_language_query() - Starting pattern analysis for {ticker_symbol}")
                        detected_patterns = detector.detect_all_patterns(ohlcv_df)

                        ticker_pattern_results = []
                        if detected_patterns:
                            logging.debug(f"DEBUG: natural_language_query() - Patterns detected for {ticker_symbol}: {len(detected_patterns)} patterns")
                            for pattern_info in detected_patterns:
                                if not pattern or pattern_info['pattern'].lower() == pattern.lower():
                                    # Create a safe copy of pattern info for response
                                    p_info = pattern_info.copy()
                                    if 'candle' in p_info:
                                        del p_info['candle'] # Remove raw candle object if present to avoid recursion/json issues

                                    ticker_pattern_results.append(p_info)
                        else:
                            logging.debug(f"DEBUG: natural_language_query() - No patterns detected for {ticker_symbol}")

                        pattern_results_all[ticker_symbol] = ticker_pattern_results

                        # Get latest candle for context display
                        latest_row = ohlcv_df.iloc[-1]
                        latest_candle = {
                            'open': float(latest_row['Open']),
                            'high': float(latest_row['High']),
                            'low': float(latest_row['Low']),
                            'close': float(latest_row['Close']),
                            'volume': int(latest_row['Volume'])
                        }

                        # Add change calculation if we have enough data
                        if len(ohlcv_df) >= 2:
                            prev_close = float(ohlcv_df.iloc[-2]['Close'])
                            change = float(latest_row['Close']) - prev_close
                            change_pct = (change / prev_close * 100.0) if prev_close != 0 else 0.0
                            latest_candle.update({
                                'previous_close': round(prev_close, 4),
                                'change': round(change, 4),
                                'change_percent': round(change_pct, 4)
                            })
                        latest_candles_all[ticker_symbol] = latest_candle

                        # Convert OHLCV DataFrame to JSON-serializable format for context (limit to last 5 candles)
                        logging.debug(f"DEBUG: natural_language_query() - Converting recent OHLCV data to JSON format for {ticker_symbol}")
                        recent_ohlcv = ohlcv_df.tail(5)
                        ohlcv_data = {str(k): v for k, v in recent_ohlcv.to_dict('index').items()}
                        ohlcv_data_all[ticker_symbol] = ohlcv_data

                        logging.debug(f"DEBUG: natural_language_query() - Pattern analysis completed for {ticker_symbol}, found {len(ticker_pattern_results)} pattern occurrences")

                    except Exception as e:
                        logging.debug(f"DEBUG: natural_language_query() - Failed to analyze patterns for {ticker_symbol}: {str(e)}")
                        failed_tickers.append(ticker_symbol)
                        continue

                if not pattern_results_all:
                    logging.debug("DEBUG: natural_language_query() - No pattern analysis available for any ticker")
                    result = jsonify({
                        'success': False,
                        'error': 'No data available',
                        'message': f'Could not analyze patterns for any of the requested tickers: {resolved_tickers}',
                        'parsed_query': parsed_result
                    }), 404
                    logging.debug("DEBUG: natural_language_query() - Returning no pattern data available response")
                    return result

                executed_actions.append('pattern_detection')
                response_data['pattern_data'] = {
                    'ohlcv_data': ohlcv_data_all,
                    'latest_candles': latest_candles_all,
                    'pattern_analysis': pattern_results_all,
                }

                # Count total patterns found across all tickers
                total_patterns = sum(len(patterns) for patterns in pattern_results_all.values())
                successful_tickers = list(pattern_results_all.keys())

                response_data['pattern_data']['patterns_found'] = total_patterns
                response_data['pattern_data']['total_tickers_analyzed'] = len(successful_tickers)

                # Create detailed per-stock summary
                stock_summaries = []
                for ticker in successful_tickers:
                    pattern_count = len(pattern_results_all[ticker])
                    if pattern_count > 0:
                        pattern_names = [p['pattern'] for p in pattern_results_all[ticker]]
                        stock_summaries.append(f"{ticker}: {pattern_count} pattern(s) ({', '.join(pattern_names)})")
                    else:
                        stock_summaries.append(f"{ticker}: No patterns")

                response_data['pattern_data']['stock_summaries'] = stock_summaries

                if len(successful_tickers) == 1:
                    # Single stock - keep existing format
                    patterns_found = len(pattern_results_all[successful_tickers[0]])
                    response_data['message'] = (
                        f"Analyzed {pattern or 'all'} patterns for {successful_tickers[0]} on {pattern_timeframe} timeframe. "
                        f"Found {patterns_found} pattern occurrence{'s' if patterns_found != 1 else ''}."
                    )
                else:
                    # Multi-stock - structured format
                    response_data['message'] = f"Analyzed {pattern or 'all'} patterns for {len(successful_tickers)} stocks on {pattern_timeframe} timeframe:\n"
                    response_data['message'] += '\n'.join(f"  • {summary}" for summary in stock_summaries)
                    response_data['message'] += f"\n\n📋 Total: {total_patterns} pattern occurrences across {len(successful_tickers)} tickers."

                if failed_tickers:
                    response_data['message'] += f" Failed to analyze: {', '.join(failed_tickers)}"

                logging.debug(f"DEBUG: natural_language_query() - Pattern detection response data prepared for {len(successful_tickers)} tickers with {total_patterns} total patterns")

            except Exception as e:
                app.logger.error(f"Pattern detection error: {traceback.format_exc()}")
                logging.debug(f"DEBUG: natural_language_query() - Exception in pattern detection: {type(e).__name__}: {str(e)}")
                result = jsonify({
                    'success': False,
                    'error': 'Pattern detection failed',
                    'message': f'Could not detect patterns: {str(e)}',
                    'parsed_query': parsed_result
                }), 500
                logging.debug("DEBUG: natural_language_query() - Returning pattern detection error response")
                return result

        # Set final action and create detailed structured summary
        if executed_actions:
            if len(executed_actions) == 1:
                response_data['action'] = executed_actions[0]
                # For single action, keep the existing message format
            else:
                response_data['action'] = 'multi_intent'
                # For multi-intent queries, replace the simple message with detailed summary
                response_data['message'] = create_detailed_multi_stock_summary(
                    executed_actions, response_data, resolved_tickers
                )
        else:
            response_data['action'] = 'no_action'
            response_data['message'] = 'No supported actions were executed for this query'

        # Log performance
        duration = (datetime.utcnow() - start_time).total_seconds()
        log_performance_metric("natural_language_query", duration)
        logging.debug(f"DEBUG: natural_language_query() - Performance logged: {duration} seconds")

        result = jsonify(response_data)
        logging.debug("DEBUG: natural_language_query() - Returning successful natural language query response")
        return result

    except Exception as e:
        app.logger.error(f"Natural language query error: {traceback.format_exc()}")
        logging.debug(f"DEBUG: natural_language_query() - Main exception caught: {type(e).__name__}: {str(e)}")
        result = jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred while processing your query'
        }), 500
        logging.debug("DEBUG: natural_language_query() - Returning main error response")
        return result

if __name__ == '__main__':
    logging.debug("DEBUG: __main__ - Script executed directly, starting Flask application")
    # Parse command line arguments for port
    port = 5000
    logging.debug("DEBUG: __main__ - Setting default port to 5000")

    if len(sys.argv) > 1 and sys.argv[1] == '--port' and len(sys.argv) > 2:
        logging.debug(f"DEBUG: __main__ - Port argument detected: {sys.argv[2]}")
        try:
            port = int(sys.argv[2])
            logging.debug(f"DEBUG: __main__ - Port successfully parsed: {port}")
        except ValueError:
            print(f"Invalid port number: {sys.argv[2]}. Using default port 5000.")
            logging.debug(f"DEBUG: __main__ - Invalid port number provided: {sys.argv[2]}, using default 5000")
            port = 5000

    logging.debug(f"DEBUG: __main__ - Starting Flask app with debug=True, host=0.0.0.0, port={port}")
    app.run(debug=True, host='0.0.0.0', port=port)
