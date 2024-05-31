import React from 'react';
import { withRouter } from 'next/router';
import { Component, RefObject } from 'react';
import Layout from '../components/Layout/MainLayout';
import { Provider } from 'react-redux';
import store from 'servDir/redux/store';
import { AuthService } from '../services';
import { LoadingPage } from 'compDir/LoadingPage';
import { GoogleTagMng } from 'servDir';
import { ToastContainer, Bounce } from 'material-react-toastify';
import 'material-react-toastify/dist/ReactToastify.css';


interface IState {
	pauseInitData: boolean
	mustLogin: boolean
}

interface IProps {
	withLayout?: boolean
}

const initWrapper = (ComponentName: React.ComponentType, acceptedRoles = []) => {
	class Content extends Component<IProps> {

		// Set default properties's values
		public static defaultProps: Partial<IProps> = {
			withLayout: true
		}

		// Set default state
		public state: IState = {
			// pauseInitData: true,
			mustLogin: false,
		}

		// NextJS init method (only available at pages folder)
		static async getInitialProps({ req, res, pathname, query, asPath, jsonPageRes, err }) {
			return {};
		}

		constructor(props) {
			super(props);
		}

		componentDidMount() {
			const queryString = window.location.search;
			const parts = queryString.split('?code=');
			const authCode = parts[1];

			if (process.env.NEXT_PUBLIC_NODE_ENV !== 'production' && authCode) {
				AuthService.verifyAuthCode(authCode, true)
					.then(({ error, authData }) => {
						// redirect to home page
						window.location.assign('/');
					});
				return;
			}

			AuthService.checkSession(true)
				.then(({ error, authData, isStopCallback }) => {
					if (error == null && !isStopCallback) {
						this.setState({
							pauseInitData: false,
							mustLogin: false
						});

					}
				})
				.catch((e) => {
					this.setState({
						pauseInitData: true,
						mustLogin: true
					});
				});
		}

		render() {

			const authReducer = store.getState().authReducer;
			const userInfo = authReducer.userInfo;
			if (authReducer.isFirstLoading) {
				return <LoadingPage message="ローディング" />;
			}

			if (this.state.mustLogin) {
				return <LoadingPage message="リダイレクト中..." />;
			}

			if (acceptedRoles.length > 0) {
				if (!acceptedRoles.includes(userInfo.roleId)) {
					return window.location.href = '/error/403';
				}
			}

			const pauseInitData = this.state.pauseInitData;
			const componentRef: RefObject<any> = React.createRef();
			return (
				<Provider store={store}>
					<ToastContainer position="top-right" autoClose={3000} transition={Bounce}
						hideProgressBar newestOnTop closeOnClick pauseOnFocusLoss draggable pauseOnHover />
					{
						this.props.withLayout ? (
							<Layout onClickMenuBtn={(callback) => {
								componentRef.current.hello();
								if (callback && typeof (callback) == 'function') {
									callback();
								}
							}}>
								<ComponentName ref={componentRef} queryData={this.props.router.query} {...this.props} {...{ pauseInitData, userInfo }} />
							</Layout>
						) : (
							<ComponentName ref={componentRef} queryData={this.props.router.query} {...this.props} {...{ pauseInitData, userInfo }} />
						)
					}
				</Provider>
			);
		}
	}
	return withRouter(Content);
};

export default initWrapper;

