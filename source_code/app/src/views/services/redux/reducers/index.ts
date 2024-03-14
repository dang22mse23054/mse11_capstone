import { combineReducers } from 'redux';

import authReducer from './auth';
import sideBarReducer from './sidebar';

const rootReducer = combineReducers({
	authReducer,
	sideBarReducer,
});

export default rootReducer;
