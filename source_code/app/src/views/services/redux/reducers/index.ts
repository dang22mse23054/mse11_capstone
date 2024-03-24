import { combineReducers } from 'redux';

import authReducer from './auth';
import sideBarReducer from './sidebar';
import videoReducer from './video';
import categoryReducer from './category';

const rootReducer = combineReducers({
	authReducer,
	sideBarReducer,
	videoReducer,
	categoryReducer,
});

export default rootReducer;
