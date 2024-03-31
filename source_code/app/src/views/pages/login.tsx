import React, { Fragment } from 'react';
import { ErrorCodes } from '../../common/constants';
import { withRouter } from 'next/router';
// import { ToastContainer, Bounce, toast } from 'material-react-toastify';
import { Grid, TextField } from '@material-ui/core';
// import MDIcon from '@mdi/react';
// import { mdiDelete, mdiFileEditOutline, mdiPlus } from '@mdi/js';
import { AuthService } from 'servDir';

import dynamic from 'next/dynamic';
const ColorButton = dynamic(
	() => import('compDir/Button'),
	{ ssr: false }
);

class Login extends React.Component {
	static getInitialProps({ req, res, pathname, query, asPath, jsonPageRes, err }) {
		const resCode = query.code;
		return { resCode };
	}

	// Set default state
	public state: IState = {
		userId: '',
		errMsg: ''
	}

	componentDidMount() {
		this.handleCode();
	}

	handleChange = (event) => {
		this.setState({
			[event.target.id]: event.target.value
		});
	}
	handleSubmit = (e) => {
		e.preventDefault();
		const userId = this.state.userId.trim();
		const regexp = /^[a-zA-Z0-9_.-]+$/;
		let errMsg = '';
		if (userId === '') {
			errMsg = 'Please enter UserID';
		} else if (!regexp.test(userId)) {
			errMsg = 'UserID is invalid format';
		}

		if (errMsg) {
			this.setState({ errMsg });
			// prevent to submit form
			return false;
		}

		AuthService.loginLocalUser(userId);
		return false;
	}

	handleCode() {
		if (this.props.resCode) {
			let errMsg = '';
			switch (Number(this.props.resCode)) {
				case ErrorCodes.NOT_EXISTED_USER:
					errMsg = 'Invalid User. Please try again!';
					break;

				case ErrorCodes.DELETED_USER:
					errMsg = 'Your account has been deleted!';
					break;

				default:
					break;
			}

			if (errMsg) {
				this.setState({ errMsg });
			}
		}

	}

	render() {
		return (
			<div className="container">
				<div className="singin-content">
					<div className="card card-signin">
						<div className="card-body">
							<h5 className="card-title text-center">
								<img style={{ width: '100%' }} src="/static/img/logo/adms.png" />
							</h5>
							<form className="form-signin" style={{ textAlign: 'center' }} method='post' onSubmit={this.handleSubmit}>
								<Grid container spacing={2} style={{ display: 'flex', flexDirection: 'column' }}>
									<Grid item className='txt-input'>
										<TextField autoComplete='off' fullWidth id="userId" placeholder="UserID" variant="filled"
											inputProps={{ style: { textAlign: 'center', borderBottom: "2px solid #6f7384", color: 'white' } }}
											error={this.state.errMsg ? true : false} helperText={this.state.errMsg}
											onChange={this.handleChange} value={this.state.userId} />
									</Grid>
									<Grid item style={{ marginTop: 50, marginBottom: 20 }}>
										<ColorButton fullWidth btnColor="indigo" btnContrast={900} variant="contained"
											onClick={this.handleSubmit}>Sign In</ColorButton>
									</Grid>
									
								</Grid>
							</form>
						</div>
					</div>
				</div>
				<style jsx>{`

                .singin-content {
                    position: absolute;
                    padding-top: 4rem;
                    width: 400px;
                    left: 50%;
                    margin-left: -200px;
                }

                .card-signin {
                    border: 0;
                    border-radius: 1rem;
                    box-shadow: 0 0.5rem 1rem 0 rgba(0, 0, 0, 0.1);
					background: #070d2b;
                }
                
                .card-signin .card-title {
					margin-bottom: 2rem;
                }
                
                .card-signin .card-body {
                    padding: 2rem;
					text-align: center;
                }

                .form-signin {
                    width: 100%;
                }
                
                .form-signin .btn {
                    font-size: 80%;
                    border-radius: 5rem;
                    letter-spacing: .1rem;
                    font-weight: bold;
                    padding: 1rem;
                    transition: all 0.2s;
                }

                .form-label-group {
                    position: relative;
                    margin-bottom: 1rem;
                    text-align: left;
                    
                }
                
                .form-label-group input {
                    border-radius: 2rem;
                    height: 3rem;
                }
                
                .btn-casso {
                    box-shadow: none !important;
                    border: 1px solid powderblue;
                    background: powderblue;
                }

                .btn-casso:hover {
                    background: lightskyblue;
                }

                .error-input {
                    background-color: #ffd5d5 !important;
                }
            `}</style>
			</div>
		);
	}


}

export default withRouter(Login);