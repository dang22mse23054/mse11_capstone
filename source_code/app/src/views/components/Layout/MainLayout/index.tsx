import { connect } from 'react-redux';
import { Actions } from 'servDir/redux/actions';
import MainLayout, { IDispatchToProps, IStateToProps } from './MainLayout';

function mapStateToProps(store): IStateToProps {
	const sideBarReducer = store.sideBarReducer;
	const authReducer = store.authReducer;

	return {
		userInfo: authReducer.userInfo,
		openSideBar: sideBarReducer.openSideBar,
	};
}

function mapDispatchToProps(dispatch, ownProps): IDispatchToProps {
	return {
		toggleSideBar: async () => {
			dispatch(Actions.SideBarAction.toggleSideBar());
		},
	};
}


export default connect(mapStateToProps, mapDispatchToProps, null, { forwardRef: true })(MainLayout);