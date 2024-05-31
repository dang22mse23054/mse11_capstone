// src/js/store
import { createStore, applyMiddleware } from 'redux';
import thunkMiddleware from 'redux-thunk';
const middlewares = [thunkMiddleware];

if (process.env.NEXT_PUBLIC_REDUX_DEBUG_LOGGER === 'true') {
	const { createLogger } = require('redux-logger');
	// createStore is the function to create Redux store 
	const loggerMiddleware = createLogger();
	middlewares.push(loggerMiddleware);
}

import rootReducer from 'servDir/redux/reducers';
const store = createStore(rootReducer, applyMiddleware(...middlewares));

export default store;

